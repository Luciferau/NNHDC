#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>


#define RESET   ""
#define GREEN   ""
#define YELLOW  ""
#define RED     ""
#define BLUE    ""
#define MAGENTA ""
#define CYAN    ""

// 如果 Windows 系统支持 ANSI 则使用这些定义
#ifdef _WIN32
#include <windows.h>
#undef RESET
#undef GREEN
#undef YELLOW
#undef RED
#undef BLUE
#undef MAGENTA
#undef CYAN

#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define RED     "\033[31m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

int supportsANSI() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return 0;
    }

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) {
        return 0;
    }

    // 启用 ENABLE_VIRTUAL_TERMINAL_PROCESSING 标志以支持 ANSI
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(hOut, dwMode)) {
        return 0;  // 无法启用 ANSI
    }

    return 1;  // 支持 ANSI
}

#else


int supportsANSI() {
    return 1;  // 在类 Unix 系统上假设支持 ANSI
}
#endif



enum LogLevel {
    ERROR_,
    INFO,
    WARN
};


const char* getColor(enum LogLevel level) {
     if (!supportsANSI()) {
        return "";  // 如果不支持 ANSI，则返回空字符串
    }
    switch (level) {
        case ERROR_: return RED; // 红色
        case INFO:  return GREEN; // 绿色
        case WARN:  return YELLOW; // 黄色
        default:    return RESET;  // 默认颜色
    }
}


void logger(enum LogLevel level, const char* format, ...) {
    
    //const char* reset = "\033[0m"; // 重置颜色
    const char* color = getColor(level); // 获取对应日志级别的颜色

    // 输出日志级别
    printf("%s", color); // 设置颜色
    if (level == ERROR_) {
        printf("[ERRO] "); // 输出 ERROR 标识
    } else {
        printf("[%s] ", (level == INFO) ? "INFO" : "WARN"); // 输出 INFO 或 WARN 标识
    }
    

     // 重置颜色并换行
    printf("%s", RESET);

    // 处理可变参数
    va_list args;
    va_start(args, format);
    vprintf(format, args); // 输出格式化的消息
    va_end(args);

   

}

 
#endif // _LOGGER_H_