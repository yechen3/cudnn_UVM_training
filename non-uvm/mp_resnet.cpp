#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <map>

#define SWITCH_CHAR             '-'
#define NUM_GPUS 2
using namespace cudl;

// TODO: move to layer.cu
Layer *residual_block(Network &model, Layer *mainline, int out_channels, int repetitions, int blockId, int &conv_name,
                     int &fbn_name, int &relu_name, int &add_name, int res_device_num) {
    Layer *shortcut = nullptr;
    int init_stride;
    int init_padding;
    int default_padding = 0;
    int default_dilation = 1;
    for (int i = 0; i < repetitions; i++) {
        init_stride = 1;
        init_padding = 1;
        shortcut = mainline;
        if (i == 0) {
            if (blockId) init_stride = 2;
            shortcut = model.add_layer(new Conv2D(shortcut, "conv2d_" + std::to_string(++conv_name), out_channels * 4, 1, init_stride, default_padding,default_dilation,res_device_num));
            shortcut = model.add_layer(new FusedBatchNormalization(shortcut, "fbn_" +  std::to_string(++fbn_name), CUDNN_BATCHNORM_SPATIAL, res_device_num));
        }
        mainline = model.add_layer(new Conv2D(mainline, "conv2d_" + std::to_string(++conv_name), out_channels, 1, 1,default_padding,default_dilation, res_device_num));
        mainline = model.add_layer(new FusedBatchNormalization(mainline, "fbn_" + std::to_string(++fbn_name), CUDNN_BATCHNORM_SPATIAL, res_device_num));
        mainline = model.add_layer(new Activation(mainline, "relu_" + std::to_string(++relu_name), CUDNN_ACTIVATION_RELU, res_device_num));

        // if (i == 0 && blockId) {
        //     init_padding = 0;
        //     mainline = model.add_layer(new Pad(mainline, "pad_", {0, 0, 0, 0, 1, 1, 1, 1}, 0));
        // }
        
        mainline = model.add_layer(new Conv2D(mainline, "conv2d_" + std::to_string(++conv_name), out_channels, 3, init_stride, init_padding,default_dilation, res_device_num));
        mainline = model.add_layer(new FusedBatchNormalization(mainline, "fbn_" + std::to_string(++fbn_name),CUDNN_BATCHNORM_SPATIAL, res_device_num));
        mainline = model.add_layer(new Activation(mainline, "relu_" + std::to_string(++relu_name),CUDNN_ACTIVATION_RELU, res_device_num));
        mainline = model.add_layer(new Conv2D(mainline, "conv2d_" + std::to_string(++conv_name), out_channels * 4, 1, 1, default_padding,default_dilation,res_device_num));
        mainline = model.add_layer(new FusedBatchNormalization(mainline, "fbn_" + std::to_string(++fbn_name), CUDNN_BATCHNORM_SPATIAL, res_device_num));

        mainline = model.add_layer(new Add(mainline, shortcut, "add_" + std::to_string(++add_name), res_device_num));
        mainline = model.add_layer(new Activation(mainline, "relu_" + std::to_string(++relu_name), CUDNN_ACTIVATION_RELU, res_device_num));
    }

    return mainline;
}

int main(int argc, char *argv[]) {
    /* configure the network */
    int batch_size_train = 20;
    int num_steps_train = 3000;
    int monitoring_step = 200;

    float lr_decay = 0.00005f;
    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps_hat = 0.00000001f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 1000;
    int num_steps_test = 10;

    int resnet_size = 50;
    int out_channels = 64;
    int conv_name = 0;
    int fbn_name = 0;
    int add_name = 0;
    int relu_name = 0;
    std::map<int,std::array<int,4>> block_size;

    // for constructing Conv2D
    int default_padding = 0;
    int default_dilation = 1;

    // for partitioning amongst multiple GPUs
    int dev_1 = 0;
    int dev_2 = 1;
    // for model->cuda(common_dev), train_data->to(cuda, common_dev), and train_target->to(cuda, common_dev)
    int common_dev = 0;

    int error = 0;
    argc -= 1;
    argv++;
    while (argc) {
        if (*argv[0] == SWITCH_CHAR) {
            switch (*(argv[0] + 1)) {
                case 'b':
                    batch_size_train = atol(argv[0] + 2);
                    break;
                case 'n':
                    num_steps_train = atol(argv[0] + 2);
                    break;
                case 'm':
                    monitoring_step = atol(argv[0] + 2);
                    break;
                case 'l':
                    load_pretrain = true;
                    break;
                case 's':
                    file_save = true;
                    break;
                case 'x':
                    batch_size_test = atol(argv[0] + 2);
                    break;
                case 'y':
                    num_steps_test = atol(argv[0] + 2);
                    break;
                case 'r':
                    resnet_size = atol(argv[0] + 2);
                    break;
                default:
                    error++;
                    break;
            }
            if (error) {
                fprintf(stderr, "Unknown switch '%c%s'\n\n", SWITCH_CHAR, argv[0] + 1);
                return error;
            }
        } else {
            fprintf(stderr, "Invalid separator '%c' for option '%s'\n\n", *argv[0], argv[0]);
            return 1;
        }
        argc -= 1;
        argv++;
    }

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, true);

    // step 2. model initialization
    Network model[NUM_GPUS];
    Layer *mainline = nullptr, *shortcut = nullptr;

    // all reside on dev_1 = 0
    // remove pad
    // mainline = model.add_layer(new Pad(mainline, "pad", {0, 0, 0, 0, 3, 3, 3, 3}, 0, dev_1)); //[1,1,28,28] -> [1,1,34,34]

    mainline = model[0].add_layer(new Conv2D(mainline, "conv2d", 64, 7, 2,default_padding,default_dilation, 0)); //[1,1,34,34] -> [1,64,14,14]
    mainline = model[0].add_layer(new FusedBatchNormalization(mainline, "fbn",CUDNN_BATCHNORM_SPATIAL, 0));
    mainline = model[0].add_layer(new Activation(mainline, "relu", CUDNN_ACTIVATION_RELU, 0)); //[1,64,14,14] -> [1,64,14,14]
    mainline = model[0].add_layer(new Pooling(mainline, "pool", 3, 1, 2, CUDNN_POOLING_MAX, 0)); //[1,64,14,14] -> [1,64,7,7]

    switch (resnet_size){
            case 18:
                block_size[18]={2, 2, 2, 2};
                break;
            case 50:
                block_size[50]={3, 6, 4, 3};
                break;
            case 101:
                block_size[101]={3, 6, 23, 3};
                break;
    }

    // all reside on dev_2 = 1
    for (int blockId = 0; blockId < 2; blockId++) {
        mainline = residual_block(model[0], mainline, out_channels, block_size[resnet_size].at(blockId), blockId, conv_name, fbn_name, relu_name, add_name, 0);
        out_channels *= 2;
    }

    for (int blockId = 2; blockId < 4; blockId++) {
        mainline = residual_block(model[1], mainline, out_channels, block_size[resnet_size].at(blockId), blockId, conv_name, fbn_name, relu_name, add_name, 1);
        out_channels *= 2;
    }

    mainline = model[1].add_layer(new Dense(mainline, "dense", 10, dev_2)); //[1,500,1,1] -> [1,10,1,1]
    mainline = model[1].add_layer(new Softmax(mainline, "softmax", dev_2));//[1,10,1,1] -> [1,10,1,1]


    // TODO: figure out which number this should be
    model[0].cuda(0);
    model[1].cuda(1);

    
    model[0].train();
    model[1].train();

    // step 3. train
    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_steps_train) {
        // nvtx profiling start
        //std::string nvtx_message = std::string("step" + std::to_string(step));
        //nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        train_data->to(cuda, 0);
        train_target->to(cuda, 1);
        
        printf("Start Forward\n");
        // forward
        Blob<float> *tmp = model[0].forward(train_data);
        tmp->cuda(1);
        model[1].forward(tmp);
        tp_count += model[1].get_accuracy(train_target);

        // back-propagation
        tmp = model[1].backward(train_target);
        tmp->cuda(0);
        model[0].backward(tmp);
        
        // update parameter
        // we will use learning rate decay to the learning rate
        //learning_rate *= 1.f / (1.f + lr_decay * step);
        model[0].update(learning_rate);
        model[1].update(learning_rate);

        // fetch next data
        step = train_data_loader.next();

        // calculation softmax loss
        if (step % monitoring_step == 0) {
            float loss = model[1].loss(train_target);
            float accuracy = 100.f * tp_count / monitoring_step / batch_size_train;

            std::cout << "step: " << std::right << std::setw(4) << step << ", loss: " << std::left << std::setw(5)
                      << std::fixed << std::setprecision(3) << loss << ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }
    }

    /*
    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    // model.train();
    model.test();

    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test) {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());
        // update shared buffer contents
        test_data->to(cuda, common_dev);
        test_target->to(cuda, common_dev);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
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