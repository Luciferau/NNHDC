/*****************************************************************************
File name: TrainHWRCNN.c
Description: CNN手写数字识别 训练代码，少部分测试代码，仅用于学习。
Author: liximing
Version: 1.0
Date:
update:2024年10月8日。
可以运转的版本。

参考文献：
1，Conv_factor/基于C语言构建CNN的手写数字识别
https://gitee.com/conv-factor/Handwritten-number-recognition

2，基于CNN的手写数字识别
https://github.com/IammyselfYBX/Handwritten-digit-recognition-based-on-CNN


***********************
******************************************************/
#include "logger.hh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <time.h>
#include <conio.h>
#include <dirent.h>
#include <fcntl.h>


 
 

// 宏定义，取两个数中的较大值
// #define max(a,b)(((a)>(b))?(a):(b))
// 定义样本数量
// 每个数字取多少个样本做为训练集。总的训练数是这个数字的10倍。
#define SAMPLE_NUM 1000
// 学习率
double lr;
// 最后通过 softmax 输出的结果
double result[10];

// 参数结构体，存储卷积核和全连接层的权重
struct parameter
{
    double conv_kernel11[3][3];
    double conv_kernel12[3][3];
    double conv_kernel21[3][3];
    double conv_kernel22[3][3];
    double conv_kernel31[3][3];
    double conv_kernel32[3][3];
    double fc_hidden_layer1[1152][180];
    double fc_hidden_layer2[180][45];
    double fc_hidden_layer3[45][10];
};
// 存储网络中每一层输出结果的尺寸
struct result
{
    double mnist_data[30][30];
    // 通道一
    double first_conv1[28][28];
    double sencond_conv1[26][26];
    double third_conv1[24][24];
    // 通道二
    double first_conv2[28][28];
    double sencond_conv2[26][26];
    double third_conv2[24][24];
    // 全连接
    double flatten_conv[1][1152];
    double first_fc[1][180];
    double first_relu[1][180];
    double second_fc[1][45];
    double second_relu[1][45];
    double outmlp[1][10]; // 全连接的输出
    double result[10];    // Softmax 的输出
};
// 保存每一张图片的结构体
struct sample
{
    double a[30][30]; // 图片数据
    int number;       // 标签
};

// 训练集结构体，训练样本 30*30
struct input
{
    double a[10][SAMPLE_NUM][30][30]; // [标签][样本数量][w][h]
};

// 保存每一张图片的结构体
struct sample Sample[SAMPLE_NUM * 10];


// liximing add 2024.10.7 ----begin
//  判断给定路径是否为目录
//  参数说明：
//  path：指向路径字符串的指针
int isDirectory(const char *path)
{
    DIR *dir = opendir(path);
    if (dir)
    {
        closedir(dir);
        return 1;
    }
    else
    {
        return 0;
    }
}

// 启用控制台的 ANSI 支持
void enableAnsiSupport() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

// 遍历目录，找到随机文件路径
void traverseDirectory(const char *directory, char **randomFilePath, int *fileCount)
{
    DIR *dir;
    struct dirent *entry;

    dir = opendir(directory);
    if (dir == NULL)
    {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
        {
            char fullPath[1024];
            snprintf(fullPath, sizeof(fullPath), "%s/%s", directory, entry->d_name);
            if (isDirectory(fullPath))
            {
                traverseDirectory(fullPath, randomFilePath, fileCount);
            }
            else
            {
                // Check if the file is a.bmp file
                const char *extension = strrchr(fullPath, '.');
                if (extension && strcmp(extension, ".bmp") == 0)
                {
                    (*fileCount)++;
                    if (rand() % (*fileCount) == 0)
                    {
                        free(*randomFilePath);
                        *randomFilePath = strdup(fullPath);
                    }
                }
            }
        }
    }

    closedir(dir);
}
// 训练前读取网络参数
// 参数说明：
BOOL read_file(struct parameter *parameter_dest)
{
    FILE *fp;
    fp = fopen("network_parameter", "rb");
    if (fp == NULL)
    {
         
        logger(WARN,"No parameter files found!\n");
        return FALSE;
    }
    struct parameter *parameter_tmp = NULL;
    parameter_tmp = (struct parameter *)malloc(sizeof(struct parameter));
    fread(parameter_tmp, sizeof(struct parameter), 1, fp);
    (*parameter_dest) = (*parameter_tmp);
    fclose(fp);
    free(parameter_tmp);
    parameter_tmp = NULL;

    return TRUE;
}

// 训练结束后保存网络参数
// 参数说明：

BOOL write_para_to_file(struct parameter *parameter_file)
{
    FILE *fp;
    fp = fopen("network_parameter", "wb");
    struct parameter *parameter_tmp;
    parameter_tmp = (struct parameter *)malloc(sizeof(struct parameter));

    (*parameter_tmp) = (*parameter_file);
    fwrite(parameter_tmp, sizeof(struct parameter), 1, fp);

    fclose(fp);
    free(parameter_tmp);
    parameter_tmp = NULL;

    return TRUE;
}
// 训练过程中的最优参数打印函数
// 参数说明：
// parameter4：指向参数结构体的指针，保存要写入文件的网络参数
void printf_file2(struct parameter *parameter4)
{
    FILE *fp;
    // 打开名为"NetworkParameters.bin"的文件，以只写二进制模式打开
    fp = fopen("NetworkParameters.bin", "wb");
    struct parameter *parameter1;
    parameter1 = (struct parameter *)malloc(sizeof(struct parameter));
    (*parameter1) = (*parameter4);
    // 将参数结构体写入文件
    fwrite(parameter1, sizeof(struct parameter), 1, fp); // 结果指针、大小、数量、文件指针
    fclose(fp);
    free(parameter1);
    parameter1 = NULL;
    return;
};

// 卷积操作，卷积核大小为 k*k
// 参数：w 和 h 分别为输入矩阵的宽度和高度，k 为卷积核大小，input_matrix 为输入矩阵，kernel 为卷积核，out_matrix 为输出矩阵
void Conv2d(int w, int h, int k, double *input_matrix, double *kernel, double *out_matrix)
{
    for (int i = 0; i < w - k + 1; i++)
        for (int j = 0; j < h - k + 1; j++)
        {
            out_matrix[i * (w - k + 1) + j] = 0;
            for (int row = i; row < i + 3; row++)
                for (int col = j; col < j + 3; col++)
                    out_matrix[i * (w - k + 1) + j] += input_matrix[row * w + col] * kernel[(row - i) * k + (col - j)];
        }
}

// 最大池化操作，池化核大小为 k*k
// 参数：w 和 h 分别为输入矩阵的宽度和高度，k 为池化核大小，input_matrix 为输入矩阵，output_matrix 为输出矩阵
void MaxPool2d(int w, int h, int k, double *input_matrix, double *output_matrix)
{
    for (int i = 0; i < w / k; i++)
        for (int j = 0; j < h / k; j++)
        {
            int max_num = -999;
            for (int row = k * i; row < k * i + k; row++)
                for (int col = k * j; col < k * j + k; col++)
                    if (input_matrix[row * w + col] > max_num)
                        max_num = input_matrix[row * w + col];
            output_matrix[i * (w / 2) + j] = max_num;
        }
}

// 用 LeakyRelu 代替 Relu，避免梯度弥散
// 参数：w 和 h 分别为输入矩阵的宽度和高度，input_matrix 为输入矩阵，output_matrix 为输出矩阵
void Relu(int w, int h, double *input_matrix, double *output_matrix)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
            output_matrix[i * w + j] = max(input_matrix[i * w + j], input_matrix[i * w + j] * 0.05);
}

// 特征图扁平化后 concat
// 参数：w 和 h 分别为输入矩阵的宽度和高度，input_matrix1 和 input_matrix2 为两个输入矩阵，output_matrix 为输出矩阵
void MatrixExtensionImproved(int w, int h, double *input_matrix1, double *input_matrix2, double *output_matrix)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
            output_matrix[i * w + j] = input_matrix1[i * w + j]; // 将通道一的特征图输出加入到 output_matrix

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
            output_matrix[w * h + i * w + j] = input_matrix2[i * w + j]; // 将通道二的特征图输出加入到 output_matrix
}

// 全连接的矩阵乘法
// 参数：w 和 h 分别为输入矩阵的宽度和高度，out_deminsion 为输出维度，input_matrix 为输入矩阵，para_layer 为参数矩阵，output_matrix 为输出矩阵
void MatrixMultiply(int w, int h, int out_deminsion, double *input_matrix, double *para_layer, double *output_matrix)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < out_deminsion; j++)
        {
            output_matrix[i * w + j] = 0;
            for (int k = 0; k < h; k++)
                output_matrix[i * w + j] += input_matrix[i * w + k] * para_layer[k * out_deminsion + j];
        }
}

// 将全连接反向传播过来的梯度拆成两部分输入到两个 channel 中
// 参数：input_matrix 为输入矩阵，splited_matrix1 和 splited_matrix2 为拆分后的矩阵
void MatrixSplit(double *input_matrix, double *splited_matrix1, double *splited_matrix2)
{
    for (int idx = 0; idx < 1152; idx++)
        if (idx < 576)
            splited_matrix1[idx] = input_matrix[idx];
        else
            splited_matrix2[idx - 576] = input_matrix[idx];
}

// 更新网络参数
// 参数：w 和 h 分别为参数矩阵的宽度和高度，input_matrix 为输入矩阵，output_matrix 为输出矩阵（更新后的参数矩阵）
void MatrixBackPropagation(int w, int h, double *input_matrix, double *output_matrix)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
            output_matrix[i * h + j] -= lr * input_matrix[i * h + j];
}

// 反向传播时的矩阵乘法
// 参数：w 和 h 分别为参数矩阵的宽度和高度，para 为参数矩阵，grad 为梯度矩阵，rgrad 为输出矩阵
void MatrixBackPropagationMultiply(int w, int h, double *para, double *grad, double *rgrad)
{
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
            rgrad[i * h + j] = para[i] * grad[j];
}

// 计算当前层的参数矩阵的梯度，利用前一层神经元梯度行矩阵乘本层神经元梯度列矩阵，得到本层参数梯度
// 参数：w 和 h 分别为参数矩阵的宽度和高度，input_matrix 为输入矩阵，grad 为梯度矩阵，output_matrix 为输出矩阵（参数梯度矩阵）
void CalculateMatrixGrad(int w, int h, double *input_matrix, double *grad, double *output_matrix)
{
    for (int i = 0; i < w; i++)
    {
        output_matrix[i] = 0; // 梯度清空，方便累加
        for (int j = 0; j < h; j++)
        {
            output_matrix[i] += input_matrix[i * h + j] * grad[j];
        }
    }
}

// 激活函数的反向传播
// 参数：w 为输入矩阵的宽度，input_matrix 为输入矩阵，grad 为梯度矩阵，output_matrix 为输出矩阵
void ReluBackPropagation(int w, double *input_matrix, double *grad, double *output_matrix)
{
    for (int i = 0; i < w; i++)
        if (input_matrix[i] > 0)
            output_matrix[i] = 1 * grad[i];
        else
            output_matrix[i] = 0.05 * grad[i];
}

// 反向传播时对梯度进行填充，由 w*h 变为(w+2*stride)*(h+2*stride)
// 参数：w 为输入矩阵的宽度，stride 为步长，input_matrix 为输入矩阵，output_matrix 为输出矩阵
void Padding(int w, int stride, double *input_matrix, double *output_matrix)
{
    for (int i = 0; i < w + 2 * stride; i++)
        for (int j = 0; j < w + 2 * stride; j++)
            output_matrix[i * (w + 2 * stride) + j] = 0; // 输出矩阵初始化
    //    for(int i=0;i<w;i++)
    //        for(int j=0;j<w;j++)
    //            output_matrix[(i+stride)*(w+2*stride)+(j+stride)]=input_matrix[i*w+j];
}

// 由于卷积核翻转 180°后恰好是导数形式，故进行翻转后与后向传播过来的梯度相乘
// 参数：k 为卷积核大小，input_matrix 为输入矩阵，output_matrix 为输出矩阵
void OverturnKernel(int k, double *input_matrix, double *output_matrix)
{
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            output_matrix[(k - 1 - i) * k + (k - 1 - j)] = input_matrix[i * k + j];
}

// 释放内存
// 参数：x 为要释放内存的指针
void MemoryFree(double *x)
{
    free(x);
    x = NULL;
}

// 使用随机数初始化网络参数
// 参数：para 为参数结构体指针
void init(struct parameter *para)
{
    srand(time(NULL));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel11[i][j] = (rand() / (RAND_MAX + 1.0));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel12[i][j] = (rand() / (RAND_MAX + 1.0));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel21[i][j] = (rand() / (RAND_MAX + 1.0)) / 5;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel22[i][j] = (rand() / (RAND_MAX + 1.0)) / 5;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel31[i][j] = (rand() / (RAND_MAX + 1.0)) / 5;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            para->conv_kernel32[i][j] = (rand() / (RAND_MAX + 1.0)) / 5;
    for (int i = 0; i < 1152; i++)
        for (int j = 0; j < 180; j++)
            para->fc_hidden_layer1[i][j] = (rand() / (RAND_MAX + 1.0)) / 1000;
    for (int i = 0; i < 180; i++)
        for (int j = 0; j < 45; j++)
            para->fc_hidden_layer2[i][j] = (rand() / (RAND_MAX + 1.0)) / 100;
    for (int i = 0; i < 45; i++)
        for (int j = 0; j < 10; j++)
            para->fc_hidden_layer3[i][j] = (rand() / (RAND_MAX + 1.0)) / 10;
}

// 前向传播，包括三层卷积，三层全连接
// 参数：input_matrix 为输入矩阵，para 为参数结构体指针，data 为结果结构体指针
void forward(double *input_matrix, struct parameter *para, struct result *data)
{
    // 第一层卷积，通道一
    Conv2d(30, 30, 3, input_matrix, &para->conv_kernel11[0][0], &data->first_conv1[0][0]);
    // 第二层卷积，通道一
    Conv2d(28, 28, 3, &data->first_conv1[0][0], &para->conv_kernel21[0][0], &data->sencond_conv1[0][0]);
    // 第三层卷积，通道一
    Conv2d(26, 26, 3, &data->sencond_conv1[0][0], &para->conv_kernel31[0][0], &data->third_conv1[0][0]); // 第一个通道得到 24*24 的特征图

    // 第一层卷积，通道二
    Conv2d(30, 30, 3, input_matrix, &para->conv_kernel12[0][0], &data->first_conv2[0][0]);
    // 第二层卷积，通道二
    Conv2d(28, 28, 3, &data->first_conv2[0][0], &para->conv_kernel22[0][0], &data->sencond_conv2[0][0]);
    // 第三层卷积，通道二
    Conv2d(26, 26, 3, &data->sencond_conv2[0][0], &para->conv_kernel32[0][0], &data->third_conv2[0][0]); // 第二个通道得到 24*24 的特征图

    // 特征图扁平化后 concat
    MatrixExtensionImproved(24, 24, &data->third_conv1[0][0], &data->third_conv2[0][0], &data->flatten_conv[0][0]);
    // 第一层全连接
    MatrixMultiply(1, 1152, 180, &data->flatten_conv[0][0], &para->fc_hidden_layer1[0][0], &data->first_fc[0][0]);
    // LeakyRelu 激活函数，第一层全连接后
    Relu(1, 180, &data->first_fc[0][0], &data->first_relu[0][0]);
    // 第二层全连接
    MatrixMultiply(1, 180, 45, &data->first_relu[0][0], &para->fc_hidden_layer2[0][0], &data->second_fc[0][0]);
    // LeakyRelu 激活函数，第二层全连接后
    Relu(1, 45, &data->second_fc[0][0], &data->second_relu[0][0]);
    // 第三层全连接
    MatrixMultiply(1, 45, 10, &data->second_relu[0][0], &para->fc_hidden_layer3[0][0], &data->outmlp[0][0]);

    double probability;
    for (int i = 0; i < 10; i++)
        probability += exp(data->outmlp[0][i]);
    for (int i = 0; i < 10; i++)
    {
        data->result[i] = exp(data->outmlp[0][i]) / probability;
        result[i] = data->result[i];
    }
    return;
}

// 反向传播，更新梯度
// 参数：label 为标签，para 为参数结构体指针，data 为结果结构体指针
void backward(int label, struct parameter *para, struct result *data)
{
    /****************************************************************************************
     * grad 结尾的变量代表每一层的梯度
     * wgrad 结尾的变量代表每一层的参数的梯度
     * rgrad 结尾的代表激活函数的梯度
     * 本网络结构是两个通道的卷积加三层全连接，每个通道有三层卷积层，无池化层，层数使用序数词标明
     ****************************************************************************************/
    int double_len = sizeof(double);
    // 分配内存，用于存储网络输出层的梯度
    double *out_grad;
    out_grad = (double *)malloc(10 * double_len); // 网络的输出是10个double类型
    // 计算交叉熵损失函数的导数，结果为 y_hat_i - y_i
    for (int i = 0; i < 10; i++)
        if (i == label)
            out_grad[i] = data->result[i] - 1;
        else
            out_grad[i] = data->result[i] - 0;
    // 三层全连接层的反向传播

    // 分配内存，用于存储第三层全连接层参数的梯度
    double *out_wgrad;
    out_wgrad = (double *)malloc(450 * double_len);
    // 进行反向传播时的矩阵乘法，计算第三层全连接层参数的梯度
    MatrixBackPropagationMultiply(45, 10, &data->second_relu[0][0], out_grad, out_wgrad);
    // 分配内存，用于存储第二层激活函数的梯度
    double *second_rgrad;
    second_rgrad = (double *)malloc(45 * double_len);
    // 计算当前层的参数矩阵的梯度，得到第二层激活函数的梯度
    CalculateMatrixGrad(45, 10, &para->fc_hidden_layer3[0][0], out_grad, second_rgrad);
    // 释放输出层梯度的内存
    MemoryFree(out_grad);
    // 分配内存，用于存储第二层全连接层的梯度
    double *second_grad;
    second_grad = (double *)malloc(180 * double_len);
    // 进行激活函数的反向传播，得到第二层全连接层的梯度
    ReluBackPropagation(45, &data->second_fc[0][0], second_rgrad, second_grad);
    // 释放第二层激活函数梯度的内存
    MemoryFree(second_rgrad);
    // 分配内存，用于存储第二层全连接层参数的梯度
    double *second_wgrad;
    second_wgrad = (double *)malloc(8100 * double_len);
    // 进行反向传播时的矩阵乘法，计算第二层全连接层参数的梯度
    MatrixBackPropagationMultiply(180, 45, &data->first_relu[0][0], second_grad, second_wgrad);
    // 分配内存，用于存储第一层激活函数的梯度
    double *first_rgrad;
    first_rgrad = (double *)malloc(180 * double_len);
    // 计算当前层的参数矩阵的梯度，得到第一层激活函数的梯度
    CalculateMatrixGrad(180, 45, &para->fc_hidden_layer2[0][0], second_grad, first_rgrad);
    // 释放第二层全连接层梯度的内存
    MemoryFree(second_grad);
    // 分配内存，用于存储第一层全连接层的梯度
    double *first_grad;
    first_grad = (double *)malloc(180 * double_len);
    // 进行激活函数的反向传播，得到第一层全连接层的梯度
    ReluBackPropagation(180, &data->first_fc[0][0], first_rgrad, first_grad);
    // 释放第一层激活函数梯度的内存
    MemoryFree(first_rgrad);
    // 分配内存，用于存储第一层全连接层参数的梯度
    double *first_wgrad;
    first_wgrad = (double *)malloc(207360 * double_len);
    // 进行反向传播时的矩阵乘法，计算第一层全连接层参数的梯度
    MatrixBackPropagationMultiply(1152, 180, &data->flatten_conv[0][0], first_grad, first_wgrad);
    // 分配内存，用于存储所有卷积层的梯度
    double *all_conv_grad;
    all_conv_grad = (double *)malloc(1152 * double_len);
    // 计算当前层的参数矩阵的梯度，得到所有卷积层的梯度
    CalculateMatrixGrad(1152, 180, &para->fc_hidden_layer1[0][0], first_grad, all_conv_grad);
    // 释放第一层全连接层梯度的内存
    MemoryFree(first_grad);

    // 以下是对通道一的反向传播计算

    // 分配内存，用于存储通道一第三层卷积层的梯度
    double *third_conv_grad1;
    third_conv_grad1 = (double *)malloc(576 * double_len);
    // 分配内存，用于存储通道一第三层卷积层的另一部分梯度
    double *third_conv_grad2;
    third_conv_grad2 = (double *)malloc(576 * double_len);
    // 将所有卷积层的梯度拆分为通道一和通道二的梯度
    MatrixSplit(all_conv_grad, third_conv_grad1, third_conv_grad2);
    // 释放所有卷积层梯度的内存
    MemoryFree(all_conv_grad);
    // 分配内存，用于存储通道一第三层卷积核的梯度
    double *third_kernel_grad;
    third_kernel_grad = (double *)malloc(9 * double_len);
    // 进行卷积操作，计算通道一第三层卷积核的梯度
    Conv2d(26, 26, 24, &data->sencond_conv1[0][0], third_conv_grad1, third_kernel_grad);
    // 分配内存，用于存储通道一第二层卷积层的梯度
    double *second_conv_grad1;
    second_conv_grad1 = (double *)malloc(676 * double_len);
    // 分配内存，用于存储通道一第三层卷积核翻转后的矩阵
    double *third_kernel_overturn;
    third_kernel_overturn = (double *)malloc(9 * double_len);
    // 将通道一第三层卷积核翻转
    OverturnKernel(3, &para->conv_kernel31[0][0], third_kernel_overturn);
    // 分配内存，用于存储通道一第三层卷积层梯度填充后的矩阵
    double *third_conv_grad_padding1;
    third_conv_grad_padding1 = (double *)malloc(784 * double_len);
    // 对通道一第三层卷积层梯度进行填充
    Padding(26, 1, third_conv_grad1, third_conv_grad_padding1);
    // 释放通道一第三层卷积层梯度的内存
    MemoryFree(third_conv_grad1);
    // 进行卷积操作，计算通道一第二层卷积层的梯度
    Conv2d(28, 28, 3, third_conv_grad_padding1, third_kernel_overturn, second_conv_grad1);
    // 释放通道一第三层卷积核翻转后的矩阵的内存
    MemoryFree(third_kernel_overturn);
    // 释放通道一第三层卷积层梯度填充后的矩阵的内存
    MemoryFree(third_conv_grad_padding1);
    // 分配内存，用于存储通道一第二层卷积核的梯度
    double *second_kernel_grad;
    second_kernel_grad = (double *)malloc(9 * double_len);
    // 进行卷积操作，计算通道一第二层卷积核的梯度
    Conv2d(28, 28, 26, &data->first_conv1[0][0], second_conv_grad1, second_kernel_grad);
    // 分配内存，用于存储通道一第一层卷积层的梯度
    double *first_conv_grad;
    first_conv_grad = (double *)malloc(784 * double_len);
    // 分配内存，用于存储通道一第二层卷积核翻转后的矩阵
    double *second_kernel_overturn;
    second_kernel_overturn = (double *)malloc(9 * double_len);
    // 将通道一第二层卷积核翻转
    OverturnKernel(3, &para->conv_kernel21[0][0], second_kernel_overturn);
    // 分配内存，用于存储通道一第二层卷积层梯度填充后的矩阵
    double *second_conv_grad_padding1;
    second_conv_grad_padding1 = (double *)malloc(900 * double_len);
    // 对通道一第二层卷积层梯度进行填充
    Padding(28, 1, second_conv_grad1, second_conv_grad_padding1);
    // 释放通道一第二层卷积层梯度的内存
    MemoryFree(second_conv_grad1);
    // 进行卷积操作，计算通道一第一层卷积层的梯度
    Conv2d(30, 30, 3, second_conv_grad_padding1, second_kernel_overturn, first_conv_grad);
    // 释放通道一第二层卷积核翻转后的矩阵的内存
    MemoryFree(second_kernel_overturn);
    // 释放通道一第二层卷积层梯度填充后的矩阵的内存
    MemoryFree(second_conv_grad_padding1);
    // 分配内存，用于存储通道一第一层卷积核的梯度
    double *first_kernel_grad;
    first_kernel_grad = (double *)malloc(9 * double_len);
    // 进行卷积操作，计算通道一第一层卷积核的梯度
    Conv2d(30, 30, 28, &data->mnist_data[0][0], first_conv_grad, first_kernel_grad);
    // 释放通道一第一层卷积层梯度的内存
    MemoryFree(first_conv_grad);

    // 以下是对通道二的反向传播计算，与通道一类似

    double *third_kernel_grad2;
    third_kernel_grad2 = (double *)malloc(9 * double_len);
    Conv2d(26, 26, 24, &data->sencond_conv2[0][0], third_conv_grad2, third_kernel_grad2);
    double *second_conv_grad2;
    second_conv_grad2 = (double *)malloc(676 * double_len);
    double *third_kernel_overturn2;
    third_kernel_overturn2 = (double *)malloc(9 * double_len);
    OverturnKernel(3, &para->conv_kernel32[0][0], third_kernel_overturn2);
    double *third_conv_grad_padding2;
    third_conv_grad_padding2 = (double *)malloc(784 * double_len);
    Padding(26, 1, third_conv_grad2, third_conv_grad_padding2);
    MemoryFree(third_conv_grad2);
    Conv2d(28, 28, 3, third_conv_grad_padding2, third_kernel_overturn2, second_conv_grad2);
    MemoryFree(third_conv_grad_padding2);
    double *second_kernel_grad2;
    second_kernel_grad2 = (double *)malloc(9 * double_len);
    Conv2d(28, 28, 26, &data->first_conv2[0][0], second_conv_grad2, second_kernel_grad2);
    double *first_conv_grad2;
    first_conv_grad2 = (double *)malloc(784 * double_len);
    double *second_kernel_overturn2;
    second_kernel_overturn2 = (double *)malloc(9 * double_len);
    OverturnKernel(3, &para->conv_kernel22, second_kernel_overturn2);
    double *second_conv_grad_padding2;
    second_conv_grad_padding2 = (double *)malloc(900 * double_len);
    Padding(28, 1, second_conv_grad2, second_conv_grad_padding2);
    MemoryFree(second_conv_grad2);
    Conv2d(30, 30, 3, second_conv_grad_padding2, second_kernel_overturn2, first_conv_grad2);
    MemoryFree(second_kernel_overturn2);
    MemoryFree(second_conv_grad_padding2);
    double *first_kernel_grad2;
    first_kernel_grad2 = (double *)malloc(9 * double_len);
    Conv2d(30, 30, 28, &data->mnist_data[0][0], first_conv_grad2, first_kernel_grad2);

    // 通道一更新参数
    MatrixBackPropagation(3, 3, first_kernel_grad, &para->conv_kernel11[0][0]);
    MatrixBackPropagation(3, 3, second_kernel_grad, &para->conv_kernel21[0][0]);
    MatrixBackPropagation(3, 3, third_kernel_grad, &para->conv_kernel31[0][0]);
    // 通道二更新参数
    MatrixBackPropagation(3, 3, first_kernel_grad2, &para->conv_kernel12[0][0]);
    MatrixBackPropagation(3, 3, second_kernel_grad2, &para->conv_kernel22[0][0]);
    MatrixBackPropagation(3, 3, third_kernel_grad2, &para->conv_kernel32[0][0]);
    // 全连接层更新参数
    MatrixBackPropagation(1152, 180, first_wgrad, &para->fc_hidden_layer1[0][0]);
    MatrixBackPropagation(180, 45, second_wgrad, &para->fc_hidden_layer2[0][0]);
    MatrixBackPropagation(45, 10, out_wgrad, &para->fc_hidden_layer3[0][0]);
    // 清空内存
    MemoryFree(first_kernel_grad);
    MemoryFree(second_kernel_grad);
    MemoryFree(third_kernel_grad);
    MemoryFree(first_kernel_grad2);
    MemoryFree(second_kernel_grad2);
    MemoryFree(third_kernel_grad2);
    MemoryFree(first_wgrad);
    MemoryFree(second_wgrad);
    MemoryFree(out_wgrad);
    return;
}

// 从图片中提取数据
// 原方法只能读取数字式文件名字。长度只有10.
int DataLoader()
{
    for (int num = 0; num < 10; num++)
    {
        for (int i = 1; i < SAMPLE_NUM; i++)
        {
            // 处理每一张照片。
            //  利用 char 型数据提取图片内容，后转化为 int 型
            char(*e);
            int(*l);
            e = (char *)malloc(sizeof(char) * 120);
            l = (int *)malloc(sizeof(int) * 960);

            char route_name1[5];
            char route_name2[30] = "Training_set\\";
            sprintf(route_name1, "%d%s", num, "\\");
            strcat(route_name2, route_name1);

            FILE *fp;
            char file_name1[10];
            sprintf(file_name1, "%d%s", i + 1, ".bmp"); // 通过 i++循环批量读取文件
            strcat(route_name2, file_name1);
            fp = fopen(route_name2, "rb");
            if (!fp) 
            {
                logger(ERROR_,"Failed to open the training set.\n");
                logger(WARN,"Please check whether the Training_set folder exists.\n");
                logger(WARN,"Please check whether the training images are complete.\n");
                return 1;
            }
            fseek(fp, 62, SEEK_SET);         // bmp 单色位图像素数据从 62 个字节处开始
            fread(e, sizeof(char), 120, fp); // 把所有数据以 char 型的格式读到 e 数组中
            fclose(fp);

            int y = 0;
            for (int r = 0; r < 120; r++)
            {
                for (int u = 1; u < 9; u++)
                {
                    l[y] = (int)(((e[r])) >> (8 - u) & 0x01); // 把每一个 char 型数据拆开成 01 数据存放到数组 l 中
                    y++;
                    if (y > 960)
                        break;
                }
            }
            int g = 0;
            for (int u = 0; u < 30; u++)
            {
                y = 0;
                for (int j = 0; j < 32; j++)
                {
                    if ((j != 30) && (j != 31))
                    {
                        Sample[num * SAMPLE_NUM + i].a[u][y] = l[g];
                        y++;
                    } // 去掉 windows 自动补 0 的数据，把真正的数据存放的样本结构体中
                    g++;
                }
            }
            int q = Sample[num * SAMPLE_NUM + i].a[0][0];
            if (q == 1)
            {
                int n = 0;
                int z = 0;
                for (int b = 0; b < 30; b++)
                {
                    n = 0;
                    for (;;)
                    {
                        if (n >= 30)
                            break;
                        if (Sample[num * SAMPLE_NUM + i].a[z][n] == 0)
                            Sample[num * SAMPLE_NUM + i].a[z][n] = 1;
                        else if (Sample[num * SAMPLE_NUM + i].a[z][n] == 1)
                            Sample[num * SAMPLE_NUM + i].a[z][n] = 0;
                        n++;
                    }
                    z++;
                }
            }
            Sample[num * SAMPLE_NUM + i].number = num; // 给样本打标签
            free(e);
            e = NULL;
            free(l);
            l = NULL;
        }
    }
    return 0;
}

// 交叉熵损失函数
// 参数说明：
// a：指向概率数组的指针
// m：标签索引
double Cross_entropy(double *a, int m)
{
    double u = 0;
    u = (-log10(a[m]));
    return u;
}

/**
 * 
 * // 定义一个名为 show_progress_bar 的函数，该函数接受两个参数：当前进度 progress 和总进度 total
void show_progress_bar(int progress, int total)
{
    // 定义进度条的宽度
    int bar_width = 50;
    // 计算当前进度的百分比
    float percent_complete = (float)progress / total;
    // 根据百分比计算进度条中的位置
    int position = bar_width * percent_complete;

    // 打印进度条的开始部分
    printf("[");
    // 使用循环打印进度条的内部
    for (int i = 0; i < bar_width; ++i)
    {
        // 如果当前位置小于进度条中的位置，打印等号
        if (i < position)
        {
            printf("=");
            // 如果当前位置等于进度条中的位置，打印大于号
        }
        else if (i == position)
        {
            printf(">");
            // 否则，打印空格
        }
        else
        {
            printf(" ");
        }
    }
    // 打印进度条的结束部分，并显示当前进度的百分比
    printf("] %d%%\r", (int)(percent_complete * 100));
    // 刷新标准输出，确保进度条立即显示
    fflush(stdout);
}
 */


void drawProgressBar(int progress, int total) {
    // 确保进度值在合理范围内
    if (progress < 0) progress = 0;
    if (progress > total) progress = total;

    // 进度条的长度
    const int barWidth = 50; // 进度条宽度
    
    float percentage = (float)progress / total; // 计算进度百分比
    int filledLength = (int)(barWidth * percentage); // 填充长度

    // 绘制进度条
    printf("\r["); // 返回到行首并开始进度条
    for (int i = 0; i < barWidth; i++) {
        if (i < filledLength) {
            printf(GREEN "="); // 填充部分用绿色
        } else {
            printf(" "); // 空白部分
        }
    }
    
    // 输出进度信息
    printf("] %.2f%%", percentage * 100); // 显示百分比

    // 完成时更改颜色
    if (progress == total) {
        printf(" " RED "    [FINISED]" RESET); // 完成状态
    } else if (percentage >= 0.5) {
        printf(" " YELLOW "[NEARING...] " RESET); // 接近完成状态
    } else  {
        printf(" [TRAINING]"); // 进行中状态
    }

    fflush(stdout); // 刷新输出缓冲区
}



// 网络训练部分，读取到图像数据进行前向传播的训练
// 参数说明：
// epochs：训练的轮数
// para：指向参数结构体的指针，存储网络参数
// data：指向结果结构体的指针，存储训练过程中的中间结果和最终结果
void train(int epochs, struct parameter *para, struct result *data)
{
    
    logger(INFO,"Train function entrance....\n");
    double corss_loss = 2; // 保存每次训练的最大交叉熵
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        lr = pow((corss_loss / 10), 1.7);
        if (lr > 0.01)
            lr = 0.01;

        drawProgressBar(epoch + 1, epochs);
        if ((epoch + 1) % 10 == 0)
        {
            fflush(stdout);
            printf("\n\t Cross entropy loss %lf  learn rate:%.10lf\n", corss_loss, lr);
            if (lr < 0.0000000001)
                printf_file2(para);
        }

        if (lr < 0.0000000001)
            printf_file2(para); // 如果找到局部最优则打印网络参数

        int a, b;
        srand(time(NULL));
        int allSamples = SAMPLE_NUM * 10;
        // allSamples =300;
        for (int q = 0; q < allSamples; q++)
        {
            // 为什么原代码中是300？难道只有300个样本吗，这是初始设定的问题。

            a = (int)((rand() / (RAND_MAX + 1.0)) * allSamples); // 确定本轮随机交换的变量下标
            b = (int)((rand() / (RAND_MAX + 1.0)) * allSamples);
            if (a >= 0 && a < allSamples && (a != b) && b >= 0 && b < allSamples)
            {
                struct sample *sample5;
                sample5 = (struct sample *)malloc(sizeof(struct sample));
                (*sample5) = Sample[a];
                Sample[a] = Sample[b];
                Sample[b] = (*sample5);
                free(sample5);
                sample5 = NULL;
            }
            else
                continue;
        }
        for (int i = 0; i < SAMPLE_NUM * 10; i++) // 训练已经打乱的所有样本
        {
            corss_loss = 0;
            struct sample *sample3;
            sample3 = (struct sample *)malloc(sizeof(struct sample));
            (*sample3) = Sample[i];
            int y = sample3->number;
            forward(&sample3->a[0][0], para, data); // 正向传播
            backward(y, para, data);
            free(sample3);
            sample3 = NULL;
            double g = Cross_entropy(&data->result[0], y); // 计算本轮最大交叉熵损失，用于指导调整学习率
            if (g > corss_loss)
                corss_loss = g;
        }
    }
    printf("\n");
    return;
}

// 用测试集中的样本测试网络，其实就是推理网络，用已训练好的网络来进行推理，检查是不是正确。
// 参数说明：
// parameter2：指向参数结构体的指针，存储网络参数
// data2：指向结果结构体的指针，存储测试过程中的中间结果和最终结果
void test_network(struct parameter *parameter2, struct result *data2)
{
    char e[120];
    int l[960];
    double data[30][30];
    // add liximing begin
    // int testcount = 1000;
    char currentDirectory[1024];
    int testCount = 10;
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == NULL)
    {
        perror("getcwd");
        return;
    }

    // 修改当前目录为 Training_set
    char trainingSetDirectory[1024];
    snprintf(trainingSetDirectory, sizeof(trainingSetDirectory), "%s/Training_set/Test_set", currentDirectory);
    if (chdir(trainingSetDirectory) != 0)
    {
        perror("chdir");
        return;
    }

    srand(time(NULL));
    char *randomFilePath = NULL;
    int fileCount = 0;

    // add by liximing end
    double RatioCorrect = 0;
    int countCroreect = 0;
    int count = 0;
    while (count < testCount)
    {
        FILE *fp;
        // char s[100];
        // sprintf(s,"%s%d%s","Training_set//Test_set//",i+1,".bmp");
        // printf("\n打开的文件名:%s\n",s);
        // 随即选中一个文件，并给出子目录名和文件名。
        traverseDirectory(trainingSetDirectory, &randomFilePath, &fileCount);
        if (randomFilePath)
        {
            const char *extension = strrchr(randomFilePath, '.');
            if (extension && strcmp(extension, ".bmp") == 0)
            {
                ;
            }
            else
            {
                continue;
            }
        }

        // 显示文件完整目录,主要是为了看，为了看一下究竟是哪个文件
        // printf("Randomly selected file full path: %s\n", randomFilePath);
        // 打开文件
        // s = randomFilePath;
        fp = fopen(randomFilePath, "rb");
        // 取数字目录。
        int parentDirectoryNumber = -1;
        char *lastSlash = strrchr(randomFilePath, '/');
        if (lastSlash)
        {
            *lastSlash = '\0';
            char subDirectory[1024];
            strcpy(subDirectory, randomFilePath);
            char *parentDirectoryName = strrchr(subDirectory, '/') + 1;
            // 显示文件上一级子目录名
            // printf("Parent directory name of the randomly selected file: %s\n", parentDirectoryName);
            parentDirectoryNumber = atoi(parentDirectoryName);
            // printf("Parent directory number: %d\n", parentDirectoryNumber);
        }

        if (fp == NULL)
        {
            printf("Cann't open the file!\n");
            //  system("pause");
            continue;
        }
        fseek(fp, 62, SEEK_SET);
        fread(e, sizeof(char), 120, fp);
        fclose(fp);
        int y = 0;
        for (int r = 0; r < 120; r++)
        {
            for (int u = 1; u < 9; u++)
            {
                l[y] = (int)((e[r]) >> (8 - u) & 0x01);
                y++;
                if (y > 960)
                    break;
            };
        };
        y = 0;
        int g = 0;
        for (int u = 0; u < 30; u++)
        {
            y = 0;
            for (int j = 0; j < 32; j++)
            {
                if ((j != 30) && (j != 31))
                {
                    data[u][y] = l[g];
                    y++;
                };
                g++;
            }
        }
        int q = data[0][0];
        if (q == 1)
        {
            int n = 0;
            int z = 0;
            for (int b = 0; b < 30; b++)
            {
                n = 0;
                for (;;)
                {
                    if (n >= 30)
                        break;
                    if (data[z][n] == 0)
                        data[z][n] = 1;
                    else if (data[z][n] == 1)
                        data[z][n] = 0;
                    n++;
                }
                z++;
            }
        }
        forward(&data[0][0], parameter2, data2); // 把获取的样本数据正向传播一次
        double sum = 0;
        int k = 0;
        for (int j = 0; j < 10; j++)
        {
            if (result[j] > sum)
            {
                sum = result[j];
                k = j; // 获取分类结果
            }
            else
                continue;
        }
        // printf("\n");
        // for(int i=0; i<10; i++) //打印分类结果
        // {
        //    printf("预测值是%d的概率：%lf\n",i,result[i]);
        // }
        count++;
        printf("all %d pics. the %d pic. real number is: %d, our prediction value is: %d %s\n", testCount, count, parentDirectoryNumber, k, parentDirectoryNumber == k ? "correct." : "wrong......");
        if (k == parentDirectoryNumber)
        {
            countCroreect++;
            // printf("最终预测值:%d\n 正确",k);
        }
        RatioCorrect = countCroreect / (count * 1.0);
    }
    printf("all %d pics,correct is %d, correct rate is:%.3f\n", count, countCroreect, RatioCorrect);
    return;
}

void welcome() {
    printf(RED   " __      __       .__                                \n" RESET);
    printf(RED   "/  \\    /  \\ ____ |  |   ____  ____   _____   ____  \n" RESET);
    printf(GREEN "\\   \\/\\/   // __ \\|  | _/ ___\\/  _ \\ /     \\_/ __ \\ \n" RESET);
    printf(GREEN " \\        /\\  ___/|  |_\\  \\__(  <_> )  Y Y  \\  ___/ \n" RESET);
    printf(BLUE  "  \\__\\/\\  /  \\___  >____/\\___  >____/|__|_|  /\\___  >\n" RESET);
    printf(BLUE  "       \\/       \\/          \\/            \\/     \\/ \n" RESET);

    printf("Welcome to use Neural network handwritten digit recognition!\n");
}
void clearBuffer() {
    int c;
    // 读取并丢弃输入缓冲区中的字符，直到遇到换行符
    while ((c = getchar()) != '\n' && c != EOF);
}

 
// liximing add 2024.10.7 ----end
int main()
{
    //Neural network handwritten digit recognition
    //system("chcp 65001");
    //system("cls");
    SetConsoleOutputCP(CP_UTF8); // 设置输出代码页为 UTF-8
    enableAnsiSupport();
    welcome();

    //while (!_kbhit()) {}
    _getch(); 

   
                                                               
    int h = DataLoader();
    if (h == 0)
        // printf("Train_set read finished\n");
        logger(INFO,"Train_set read finished\n");
    else if (h == 1)
    {
        //printf("no train set, quit\n");
        logger(ERROR_,"No 'Train_set' quit...\n");
        _getch();
        //system("pause");
        return 0;
    }
    struct parameter *storage;                                        // 定义存放网络参数的结构体
    (storage) = (struct parameter *)malloc(sizeof(struct parameter)); // 动态分配空间
    struct result *data;
    (data) = (struct result *)malloc(sizeof(struct result));

   // printf("being to raad former para.... \n");
    logger(INFO,"Being to read former paramter...\n");
    BOOL bh = read_file(storage);
    if (!bh)
    {
         
        logger(INFO,"No parameters, start to initianlize para...\n");
        init(storage);
        // printf_file(storage);
        
        logger(INFO,"Parameters initializeied!\n");
    }
    else
        //printf("para read OK!\n");
        logger(INFO,"Parameters readed!\n");
    printf("\033[01;32minput train times:\033[0;0m  ");
    
    int epoch;
    scanf("%d", &epoch);
    if (epoch > 10000)
    {
        epoch = 100;
    }
   
    //printf("\033[01;32mStart train...\033[0;0m\n");
    printf("\tStart train...\n");
    train(epoch, storage, data);
    write_para_to_file(storage);
    test_network(storage, data);
    _getch(); 
    return 0;
}
