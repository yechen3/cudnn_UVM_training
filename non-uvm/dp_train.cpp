#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <cuda_runtime.h>
#include "nccl.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>

using namespace cudl;

#define NUM_GPUS 2

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                             \
  if (e!= cudaSuccess) {                            \
    printf("Failed, Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
    /* configure the network */
    int batch_size_train = 1;
    int num_steps_train = 10;
    int monitoring_step = 1;

    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    int device_num = 1;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, false);

    Network model[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        // step 2. model initialization
        Layer *mainline = nullptr;
        mainline = model[i].add_layer(new Dense(mainline, "dense1", i, 1000));
        mainline = model[i].add_layer(new Activation(mainline, "relu", i, CUDNN_ACTIVATION_RELU));
        mainline = model[i].add_layer(new Dense(mainline, "dense2", i, 10));
        mainline = model[i].add_layer(new Softmax(mainline, "softmax", i));
        model[i].cuda(i);

        if (load_pretrain)
            model[i].load_pretrain();
        model[i].train();
    }

    //initializing NCCL
    ncclComm_t comms[NUM_GPUS];

    int nDev = NUM_GPUS;
    int size = 32*1024*1024;
    int devs[NUM_GPUS];

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    for (int i = 0; i < nDev; ++i) {
        devs[i] = i;
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
    }

    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // step 3. train
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    int step = 0;
    while (step < num_steps_train)
    {
        // nvtx profiling start
        //std::string nvtx_message = std::string("step" + std::to_string(step));
        //nvtxRangePushA(nvtx_message.c_str());

        for (int i = 0; i < NUM_GPUS; i++) {
             // update shared buffer contents
            train_data->to(cuda, i);
            train_target->to(cuda, i);
            
            // forward
            model[i].forward(train_data);
            tp_count += model[i].get_accuracy(train_target);

            // back-propagation
            model[i].backward(train_target);

            // update parameter
            // we will use learning rate decay to the learning rate
            learning_rate *= 1.f / (1.f + lr_decay * step);
            model[i].update(learning_rate);
            
            // calculation softmax loss
            if (step % monitoring_step == 0)
            {
                float loss = model[i].loss(train_target);
                float accuracy =  100.f * tp_count / monitoring_step / batch_size_train;
                
                std::cout << "step: " << std::right << std::setw(4) << step << \
                            ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                            ", accuracy: " << accuracy << "%" << std::endl;

                tp_count = 0;
            }
            
            // fetch next data
            step = train_data_loader.next();
        }

        nvtxRangeId_t id1 = nvtxRangeStartA("NCCL Communication");
        for (int l = 0; l < model[0].layers().size(); l++) {
            for (int i = 0; i < nDev; ++i) {
                Layer * layer = model[i].layers()[l];
                if (layer->has_weight()) {
                    const float * w = layer->weights()->cuda(i);
                    int w_size = layer->weights()->size();
                    cudaMemcpy(sendbuff[i], w, w_size, cudaMemcpyDeviceToDevice);
                }

                if (layer->has_biases()) {
                    const float * b = layer->biases()->cuda(i);
                    int b_size = layer->biases()->size();
                    if (layer->has_weight()) {
                        int w_size = layer->weights()->size();
                        cudaMemcpy(sendbuff[i]+w_size*sizeof(float), b, b_size, cudaMemcpyDeviceToDevice);
                    }
                    else
                        cudaMemcpy(sendbuff[i], b, b_size, cudaMemcpyDeviceToDevice);
                }               
            }

            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nDev; ++i) {
                NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclAvg,
                    comms[i], s[i]));
            }
            NCCLCHECK(ncclGroupEnd());

            for (int i = 0; i < nDev; ++i) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaStreamSynchronize(s[i]));

                Layer * layer = model[i].layers()[l];
                if (layer->has_weight()) {
                    float * w = layer->weights()->cuda(i);
                    int w_size = layer->weights()->size();
                    cudaMemcpy(w, recvbuff[i], w_size, cudaMemcpyDeviceToDevice);
                }

                if (layer->has_biases()) {
                    float * b = layer->biases()->cuda(i);
                    int b_size = layer->biases()->size();
                    if (layer->has_weight()) {
                        int w_size = layer->weights()->size();
                        cudaMemcpy(b, recvbuff[i]+w_size*sizeof(float), b_size, cudaMemcpyDeviceToDevice);
                    }
                    else
                        cudaMemcpy(b, recvbuff[i], b_size, cudaMemcpyDeviceToDevice);

                    //printf("Yechen :: memcpy bias back to %p\n", b);
                }                              
            }
        }
        nvtxRangeEnd(id1);

        // nvtx profiling end
        //nvtxRangePop();
    }

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    
    /*
    model.add_layer(new Conv2D("conv1", 20, 5));
    model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Conv2D("conv2", 50, 5));
    model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Dense("dense1", 500));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new Softmax("softmax"));
    */

    
/*
    // trained parameter save
    if (file_save)
        model[0].write_file();

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model[0].test();
    
    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
		test_data->to(cuda, device_num);
		test_target->to(cuda, device_num);

        // forward
        model[0].forward(test_data);
        tp_count += model[0].get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }

    // step 4. calculate loss and accuracy
    float loss = model[0].loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;
*/
    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
