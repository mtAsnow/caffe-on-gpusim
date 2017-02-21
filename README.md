# caffe-on-gpusim
run the caffe on the gpusim simulator


安装Ubuntu环境及gpusim和caffe

一  环境安装
1.安装ubuntu14.04.4 LTS 64位
	Kernel为4.2.0-42-generic
http://mirrors.163.com/ubuntu-releases/14.04/
注：其他14.04 版系统应该也可以，启动U盘可用LILI USB creator 制作，ARM公司电脑无法使用UltraISO。
编译器为系统自带，gcc4.8.4
2.安装cuda7.5sdk   （Gpusim最高支持cuda7.5）
https://developer.nvidia.com/cuda-75-downloads-archive
注意：安装时需要关闭图形窗口。最好屏蔽集成显卡。
安装文档：
http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf
也可谷歌其他人的安装经验。
3.安装gpusim
注意：需安装dev版本，master版不支持cuda7.5
https://github.com/gpgpu-sim/gpgpu-sim_distribution/tree/dev
安装方法及依赖见README。
4.安装caffe
http://caffe.berkeleyvision.org/
https://github.com/BVLC/caffe 
master版17年2月
安装方法及依赖见在线文档。

二  gpusim及caffe的修改
主要修改内容：1. 将gpusim的ptx文件解析路径连到libcaffe.so                2.  用cuda代码重写caffe内对cublas.h库调用的函数                 注：重写后的运行时间约为原时间的1.2倍

安装方法：将float_newcublas.cu和double_newcublas.cu拷贝到/#your/#path/caffe-master/include/caffe/  下
                      将math_functions.cu拷贝到/#your/#path/caffe-master/src/caffe/util/   下并覆盖原文件
                      在caffe-master目录下执行make clean 然后执行 make all。
                      将cuda_runtime_api.cc拷贝到/#your/#path/gpu-sim/libcuda/   下并覆盖原文件。
                      在gpu-sim目录下执行make clean 然后执行 make all。
详细修改内容：
1．在/#your/#path/gpu-sim/libcuda/cuda_runtime_api.cc 中的get_app_binary()函数中将char self_exe_path[1025] 改为caffe中libcaffe.so的路径。
2.重写 /#your/#path/caffe-master/src/caffe/util/math_functions.cu
中的 gemm(), gemv(), axpy(), scal(), dot(), asum(), scale() 
具体功能如下：
gemm 函数
功能： C=alpha*A*B+beta*C 
gemv 函数
功能： y=alpha*A*x+beta*y 
axpy 函数
功能： Y=alpha*X+Y 
scal 函数
功能：X = alpha*X 
dot 函数
功能： 返回 vector X 和 vector Y 的内积
asum 函数
功能：计算 vector x 的所有element的绝对值之和
scale 函数
功能：y = alpha*x
注：A, B, C 为矩阵， x, y 为向量， alpha, beta 为常数。
gemm, gemv 函数参数有转置参数需注意。转置的目的在于区分矩阵存储的方式为行主序或列主序。

三 测试运行
更改/#your/#path/caffe-master/examples/mnist/lenet_solver.prototxt   中配置参数其中：
Test_iter:1
Test_interval:1
Display:1
Max_iter:1
更改/#your/#path/caffe-master/examples/mnist/lenet_train_test.prototxt   中配置参数
将里面4个num_output 改为2。

在caffe-master 目录下运行：
#sh data/mnist/get_mnist.sh
#sh examples/mnist/create_mnist.sh
#sh examples/mnist/train_lenet.sh

运行完成需约2小时，由于更改了配置参数，结果并不准确。 
