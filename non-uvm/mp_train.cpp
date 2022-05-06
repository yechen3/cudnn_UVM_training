#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <map>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

using namespace cudl;

#define NUM_GPUS 2
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

Network model[NUM_GPUS];

std::vector<std::vector<Blob<float> *>> input;
std::vector<std::vector<Blob<float> *>> grad_input;

std::vector<std::mutex> m (NUM_GPUS);
std::vector<std::condition_variable> cv (NUM_GPUS);
std::vector<bool> input_ready (NUM_GPUS, false);
std::vector<bool> grad_input_ready (NUM_GPUS, false);

std::mutex global_m;
std::condition_variable global_cv;
bool forward = false;
bool backward = false;
std::mutex ready_m;
std::mutex done_m;
std::condition_variable ready_cv;
std::condition_variable done_cv;
bool ready = false;
bool done = false;

std::vector<std::thread> ThreadVector;

void worker_thread(int device_num, int steps) {
    std::unique_lock<std::mutex> lk(m[device_num], std::defer_lock);
    std::unique_lock<std::mutex> g_lk(global_m, std::defer_lock);
    int iter = 0;
    while (iter < steps) {
        // forward
        if (device_num == 0) {
            g_lk.lock();
            global_cv.wait(g_lk, []{return ready;});
            g_lk.unlock();
        }

        for (int i = 0; i < NUM_GPUS; i++) {
            // Every worker except first worker waits for the previous to finish
            if (device_num != 0) {
                lk.lock();
                cv[device_num].wait(lk, [=]{return input_ready[device_num];});
                lk.unlock();
            }
            input[device_num][i]->cuda(device_num);
            if (device_num != NUM_GPUS - 1) {
                std::unique_lock<std::mutex> next_lk(m[device_num + 1], std::defer_lock);
                lk.lock();
                next_lk.lock();
                input[device_num + 1][i] = model[device_num].forward(input[device_num][i]);
                input_ready[device_num+1] = true;
                input_ready[device_num] = false;
                lk.unlock();
                next_lk.unlock();
                cv[device_num + 1].notify_one();
            } else {
                model[device_num].forward(input[device_num][i]);
            }
        }
        if (device_num != NUM_GPUS - 1) {
            g_lk.lock();
            global_cv.wait(g_lk, []{return forward;});
            g_lk.unlock();
        } else {
            g_lk.lock();
            forward = true;
            backward = false;
            g_lk.unlock();
            global_cv.notify_all();
        }

        // backward
        for (int i = 0; i < NUM_GPUS; i++) {
            // Every worker except first worker waits for the previous to finish
            if (device_num != NUM_GPUS - 1) {
                lk.lock();
                cv[device_num].wait(lk, [=]{return grad_input_ready[device_num];});
                lk.unlock();
            }
            grad_input[device_num][i]->cuda(device_num);
            if (device_num != 0) {
                std::unique_lock<std::mutex> prev_lk(m[device_num - 1], std::defer_lock);
                prev_lk.lock();
                lk.lock();
                grad_input[device_num - 1][i] = model[device_num].backward(grad_input[device_num][i]);
                grad_input_ready[device_num - 1] = true;
                grad_input_ready[device_num] = false;
                lk.unlock();
                prev_lk.unlock();
                cv[device_num - 1].notify_one();
            } else {
                model[device_num].backward(grad_input[device_num][i]);
            }
        }

        if (device_num != 0) {
            g_lk.lock();
            global_cv.wait(g_lk, []{return backward;});
            g_lk.unlock();
        } else {
            g_lk.lock();
            backward = true;
            forward = false;
            done = true;
            ready = false;
            g_lk.unlock();
            global_cv.notify_all();
        }

        iter++;
    }

    //cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, false);

    // step 2. model initialization
    Layer *mainline = nullptr;
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
    mainline = model[0].add_layer(new Dense(mainline, "dense1", 0, 1000));
    mainline = model[0].add_layer(new Activation(mainline, "relu", 0, CUDNN_ACTIVATION_RELU));
    model[0].cuda(0);
    mainline = model[1].add_layer(new Dense(mainline, "dense2", 1, 10));
    mainline = model[1].add_layer(new Softmax(mainline, "softmax", 1));
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

    std::unique_lock<std::mutex> g_lk(global_m, std::defer_lock);

    for (int i=0; i < NUM_GPUS; i++) {
        std::vector<Blob<float> *> v1 (NUM_GPUS, nullptr);
        input.push_back(v1);
        grad_input.push_back(v1);
    }

    for (int i=0; i < NUM_GPUS; i++) {
        ThreadVector.emplace_back([=](){worker_thread(i, num_steps_train/NUM_GPUS);});
    }
    
    while (step < num_steps_train)
    {
        // nvtx profiling start
        //std::string nvtx_message = std::string("step" + std::to_string(step));
        //nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        for (int i=0; i < NUM_GPUS; i++) {
            input[0][i] = train_data;
            input[0][i]->to(cuda, 0);
            grad_input[NUM_GPUS-1][i] = train_target;
            grad_input[NUM_GPUS-1][i]->to(cuda, NUM_GPUS-1);

            step = train_data_loader.next();
        }

        g_lk.lock();
        ready = true;
        done = false;
        g_lk.unlock();
        global_cv.notify_all();
        
        g_lk.lock();
        global_cv.wait(g_lk, []{return done;});
        g_lk.unlock();

        for (int i=0; i < NUM_GPUS; i++) {
            tp_count += model[NUM_GPUS - 1].get_accuracy(grad_input[NUM_GPUS - 1][i]);
            //cudaDeviceSynchronize();
        }

        // update parameter
        // we will use learning rate decay to the learning rate
        learning_rate *= 1.f / (1.f + lr_decay * step);
        for (int i=0; i < NUM_GPUS; i++) {
            model[i].update(learning_rate);
        }
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
    }

    for(auto& t: ThreadVector) {
            t.join();
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
