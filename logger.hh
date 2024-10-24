#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>


#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define RED     "\033[31m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"


enum LogLevel {
    ERROR_,
    INFO,
    WARN
};


const char* getColor(enum LogLevel level) {
    switch (level) {
        case ERROR_: return RED; // 红色
        case INFO:  return GREEN; // 绿色
        case WARN:  return YELLOW; // 黄色
        default:    return RESET;  // 默认颜色
    }
}


void logger(enum LogLevel level, const char* format, ...) {
    
    const char* reset = "\033[0m"; // 重置颜色
    const char* color = getColor(level); // 获取对应日志级别的颜色

    // 输出日志级别
    printf("%s", color); // 设置颜色
    if (level == ERROR_) {
        printf("[ERROR] "); // 输出 ERROR 标识
    } else {
        printf("[%s]  ", (level == INFO) ? "INFO" : "WARN"); // 输出 INFO 或 WARN 标识
    }
    

     // 重置颜色并换行
    printf("%s", reset);

    // 处理可变参数
    va_list args;
    va_start(args, format);
    vprintf(format, args); // 输出格式化的消息
    va_end(args);

   

}

 
#endif // _LOGGER_H_