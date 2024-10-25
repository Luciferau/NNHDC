/*****************************************************************************
File name: Vonepic.c
Description: CNN手写数字识别的单图片识别程序，需要在训练好的网络参数文件 network_parameter 。
采用 C 语言编写，基于 CNN 模型，使用 MNIST 手写数字数据集进行训练和测试。

Author: liximing
Version: 1.0
Date:
update:2024年10月8日。


参考文献：
1，Conv_factor/基于C语言构建CNN的手写数字识别
https://gitee.com/conv-factor/Handwritten-number-recognition

2，基于CNN的手写数字识别
https://github.com/IammyselfYBX/Handwritten-digit-recognition-based-on-CNN


***********************
******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <time.h>
#include <conio.h>
#include <dirent.h>
#include <fcntl.h>
#include "logger.hh"

// 定义一个宏 TESTCUNT，其值为 1000
#define TESTCUNT 1000

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

// 宏定义，取两个数中的较大值
// #define max(a,b)(((a)>(b))?(a):(b))
// 定义样本数量
// 每个数字取多少个样本做为训练集。
#define SAMPLE_NUM 500
// 学习率
double lr;
// 最后通过 softmax 输出的结果
double result[10];

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
        //printf("File opening failed, please check whether the network parameters file is in the training set folder!\n");
        logger(ERROR_,"network_parameter opening failed.\n");
        logger(WARN,"Please check whether the network parameters file is in the training set folder!\n");
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

// 用测试集中的样本测试网络，其实就是推理网络，用已训练好的网络来进行推理，检查是不是正确。
// 参数说明：
// parameter2：指向参数结构体的指针，存储网络参数
// data2：指向结果结构体的指针，存储测试过程中的中间结果和最终结果
int test_onePIC(struct parameter *parameter2, struct result *data2, char *FileName)
{
    char e[120];
    int l[960];
    double data[30][30];
    // add liximing begin


    FILE *fp;
    // char s[100];
    // sprintf(s,"%s%d%s","Training_set//Test_set//",i+1,".bmp");
    
    logger(INFO,"Open file name: %s\n",FileName);
    // 随即选中一个文件，并给出子目录名和文件名。

    // 显示文件完整目录,主要是为了看，为了看一下究竟是哪个文件
    // printf("Randomly selected file full path: %s\n", randomFilePath);
    // 打开文件
    // s = randomFilePath;
    fp = fopen(FileName, "rb");
    // 取数字目录。

    if (fp == NULL)
    {
        //printf("Cann't open the file!\n");
        logger(ERROR_,"Cann't open the file %s!\n",FileName);
        return -1;
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
    printf("\n");
    for (int i = 0; i < 10; i++) // 打印分类结果
    {
        printf("Prediction value is %d, The probability is: %lf\n", i, result[i]);
    }
    return k;
}
void enableAnsiSupport() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}



void welcome() {
    printf(RED   " __      __       .__                                \n" RESET);
    printf(RED   "/  \\    /  \\ ____ |  |   ____  ____   _____   ____  \n" RESET);
    printf(GREEN "\\   \\/\\/   // __ \\|  | _/ ___\\/  _ \\ /     \\_/ __ \\ \n" RESET);
    printf(GREEN " \\        /\\  ___/|  |_\\  \\__(  <_> )  Y Y  \\  ___/ \n" RESET);
    printf(BLUE  "  \\__\\/\\  /  \\___  >____/\\___  >____/|__|_|  /\\___  >\n" RESET);
    printf(BLUE  "       \\/       \\/          \\/            \\/     \\/ \n" RESET);

    printf("Welcome to use Neural network handwritten digit recognition Test program!\n");
}

int main()
{
    SetConsoleOutputCP(CP_UTF8); // 设置输出代码页为 UTF-8
    enableAnsiSupport();
    welcome();
    struct parameter *storage;                                        // 定义存放网络参数的结构体
    (storage) = (struct parameter *)malloc(sizeof(struct parameter)); // 动态分配空间
    struct result *data;
    (data) = (struct result *)malloc(sizeof(struct result));

   
    logger(INFO,"Reading former parameter input......\n");
    BOOL h = read_file(storage);
    if (!h)
    {
        
        logger(ERROR_,"No parameters,Please to initianlize para!\n");
        logger(ERROR_,"Test program will quit...\n");
        _getch();
        return 0;
    }
    else
    {
        logger(INFO,"Reading former parameter input success!\n");
    }
    
    char s[100];

    int i = 1;
    while (1){
        logger(INFO,"Please input the file name:(q to quit\n");
        scanf("%s", s);
        if(s[0] == 'q' || s[0] == 'Q'){
            break;
        }
        int k = test_onePIC(storage, data, s);
        logger(INFO,"The prediction value is %d\n\n", k);
        

    }

    // sprintf(s,"%s","1.bmp");   
    // int k = test_onePIC(storage, data, s);
    // printf("Prediction value is %d\n", k);

    // sprintf(s,"%s","8.bmp");   
    // k = test_onePIC(storage, data, s);
    // printf("Prediction value is %d\n", k);

    // sprintf(s,"%s","9.bmp");   
    // k = test_onePIC(storage, data, s);
    // printf("Prediction value is %d\n", k);
     _getch();

    return 0;
}