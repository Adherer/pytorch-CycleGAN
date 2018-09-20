import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()         # opt用于处理命令行参数
    data_loader = CreateDataLoader(opt)     # data_loader用于加载数据
    dataset = data_loader.load_data()       # 加载数据集
    dataset_size = len(data_loader)         # 数据集的size
    print('#training images = %d' % dataset_size)     # training data:1096张(train, trainA, trainB均为1096张)

    model = create_model(opt)           # 创建模型，opt.model默认值是cycle_gan
    model.setup(opt)                    # 模型读取opt中的参数，并进行相关初始化操作
    visualizer = Visualizer(opt)        # 用于可视化输出
    total_steps = 0

    # 训练200个epoch
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:   # print_freq = 100
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size      # default batch_size = 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()       # 优化参数

            if total_steps % opt.display_freq == 0:     # display_freq = 400
                save_result = total_steps % opt.update_html_freq == 0    # update_html_freq = 1000
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size

                '''
                一个输出的sample：(epoch: 1, iters: 100, time: 0.328, data: 0.209) 
                D_A: 0.288 G_A: 0.167 cycle_A: 1.705 idt_A: 0.949 D_B: 0.461 G_B: 1.016 cycle_B: 2.035 idt_B: 0.593 
                '''
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:      # save_latest_freq = 5000
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:   # save_epoch_freq = 5
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
