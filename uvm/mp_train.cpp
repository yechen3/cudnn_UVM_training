#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>

using namespace cudl;

#define NUM_GPUS 2

int main(int argc, char* argv[])
{
    /* configure the network */
    int batch_size_train = 32;
    int num_steps_train = 100;
    int monitoring_step = 10;

    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    int device_0 = 0;
    int device_1 = 1;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, false);

    // step 2. model initialization
    Network model[NUM_GPUS];
    Layer *mainlain = nullptr;
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
    //mainlain = model[0].add_layer(new Conv2D(mainlain, "conv1", device_0, 20, 5));
    //mainlain = model[0].add_layer(new Pooling(mainlain, "pool", device_0, 2, 0, 2, CUDNN_POOLING_MAX));
    //mainlain = model[0].add_layer(new Conv2D(mainlain, "conv2", device_0, 50, 5));
    //mainlain = model[0].add_layer(new Pooling(mainlain, "pool", device_0, 2, 0, 2, CUDNN_POOLING_MAX));
    
    mainlain = model[0].add_layer(new Dense(mainlain, "dense1", device_0, 1000));
    mainlain = model[0].add_layer(new Activation(mainlain, "relu", device_0, CUDNN_ACTIVATION_RELU));
    model[0].cuda(0);
    mainlain = model[1].add_layer(new Dense(mainlain, "dense2", device_1, 10));
    mainlain = model[1].add_layer(new Softmax(mainlain, "softmax", device_1));
    model[1].cuda(1);

    //if (load_pretrain)
    //    model.load_pretrain();
    model[0].train();
    model[1].train();

    // step 3. train
    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_steps_train)
    {
        // nvtx profiling start
        //std::string nvtx_message = std::string("step" + std::to_string(step));
        //nvtxRangePushA(nvtx_message.c_str());
        
        // update shared buffer contents
        train_data->to(cuda, 0);
        train_target->to(cuda, 1);

        // forward
        Blob<float> *tmp = model[0].forward(train_data);
        cudaDeviceSynchronize();
        model[1].forward(tmp);
        cudaDeviceSynchronize();
        tp_count += model[1].get_accuracy(train_target);
        cudaDeviceSynchronize();

        // back-propagation
        tmp = model[1].backward(train_target);
        cudaDeviceSynchronize();
        model[0].backward(tmp);
        cudaDeviceSynchronize();
        
        // update parameter
        // we will use learning rate decay to the learning rate
        learning_rate *= 1.f / (1.f + lr_decay * step);
        model[0].update(learning_rate);
        model[1].update(learning_rate);
        cudaDeviceSynchronize();

        // nvtx profiling end
        //nvtxRangePop();

        // calculation softmax loss
        if (step % monitoring_step == 0)
        {
            float loss = model[1].loss(train_target);
            float accuracy =  100.f * tp_count / monitoring_step / batch_size_train;
            
            std::cout << "step: " << std::right << std::setw(4) << step << \
                         ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }

        // fetch next data
        step = train_data_loader.next();
    }

    // trained parameter save
    //if (file_save)
    //    model.write_file();
    /*
    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model.test();
    
    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test)
    {
        // nvtx profiling start
        //std::string nvtx_message = std::string("step" + std::to_string(step));
        //nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
		test_data->to(cuda, device_num);
		test_target->to(cuda, device_num);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        //nvtxRangePop();
    }

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;
    */
    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
