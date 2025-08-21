#ifndef NNL2_LOG_H
#define NNL2_LOG_H

#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>

// ANSI Colors
#define NNL2_COLOR_RESET   "\x1b[0m"
#define NNL2_COLOR_DEBUG   "\x1b[36m"  // Cyan
#define NNL2_COLOR_INFO    "\x1b[32m"  // Green
#define NNL2_COLOR_WARNING "\x1b[33m"  // Yellow
#define NNL2_COLOR_ERROR   "\x1b[31m"  // Red
#define NNL2_COLOR_FATAL   "\x1b[35m"  // Magenta

// Text constants
#define NNL2_LOG_PREFIX_DEBUG   "DEBUG"
#define NNL2_LOG_PREFIX_INFO    "INFO"
#define NNL2_LOG_PREFIX_WARNING "WARN"
#define NNL2_LOG_PREFIX_ERROR   "ERROR"
#define NNL2_LOG_PREFIX_FATAL   "FATAL"

#define NNL2_LOG_TIMESTAMP_FORMAT "%H:%M:%S"
#define NNL2_LOG_TIMESTAMP_SIZE   20

#define NNL2_LOG_SYSTEM_ERROR_PREFIX " (system error "
#define NNL2_LOG_SYSTEM_ERROR_SUFFIX ": "
#define NNL2_LOG_SYSTEM_ERROR_END    ")"

// Logging levels
typedef enum {
    NNL2_LOG_LEVEL_DEBUG,
    NNL2_LOG_LEVEL_INFO,
    NNL2_LOG_LEVEL_WARNING,
    NNL2_LOG_LEVEL_ERROR,
    NNL2_LOG_LEVEL_FATAL
} nnl2_log_level_t;

// Configuration Structure
typedef struct {
    int enable_color;
    int enable_timestamps;
    int enable_debug_info;
    nnl2_log_level_t min_log_level; 
} nnl2_log_config_t;

// Global variables
extern nnl2_log_config_t nnl2_log_current_config;

// Configuration Functions
void nnl2_log_init(int enable_color, int enable_timestamps, int enable_debug_info, nnl2_log_level_t min_log_level);
void nnl2_log_set_color(int enabled);
void nnl2_log_set_timestamps(int enabled);
void nnl2_log_set_debug_info(int enabled);
void nnl2_log_set_min_level(nnl2_log_level_t min_level);
void nnl2_log_get_config(nnl2_log_config_t* out_config);

// Main logging function
void nnl2_log(nnl2_log_level_t level, const char* file, int line, const char* func, const char* format, ...);

// Macros for logging
#define NNL2_LOG(level, ...) nnl2_log(level, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define NNL2_DEBUG(...)      NNL2_LOG(NNL2_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define NNL2_INFO(...)       NNL2_LOG(NNL2_LOG_LEVEL_INFO, __VA_ARGS__)
#define NNL2_WARN(...)       NNL2_LOG(NNL2_LOG_LEVEL_WARNING, __VA_ARGS__)
#define NNL2_ERROR(...)      NNL2_LOG(NNL2_LOG_LEVEL_ERROR, __VA_ARGS__)
#define NNL2_FATAL(...)      NNL2_LOG(NNL2_LOG_LEVEL_FATAL, __VA_ARGS__)

// Standard parameters for initialization
#define NNL2_LOG_DEFAULT_COLOR        1
#define NNL2_LOG_DEFAULT_TIMESTAMPS   1
#define NNL2_LOG_DEFAULT_DEBUG_INFO   1
#define NNL2_LOG_DEFAULT_MIN_LEVEL    NNL2_LOG_LEVEL_INFO

#define NNL2_FUNC_ENTER() NNL2_DEBUG("Function %s() entered", __func__)
#define NNL2_FUNC_EXIT()  NNL2_DEBUG("Function %s() exited", __func__)

// --------------- Realization ---------------

// Global configuration
nnl2_log_config_t nnl2_log_current_config = {
    .enable_color = 1,
    .enable_timestamps = 1,
    .enable_debug_info = 0,
    .min_log_level = NNL2_LOG_LEVEL_DEBUG  // By default all levels are
};

// Configuration functions

void nnl2_log_init(int enable_color, int enable_timestamps, int enable_debug_info, nnl2_log_level_t min_log_level) {
    nnl2_log_current_config.enable_color = enable_color;
    nnl2_log_current_config.enable_timestamps = enable_timestamps;
    nnl2_log_current_config.enable_debug_info = enable_debug_info;
    nnl2_log_current_config.min_log_level = min_log_level;
}

void nnl2_log_set_color(int enabled) {
    nnl2_log_current_config.enable_color = enabled;
}

void nnl2_log_set_timestamps(int enabled) {
    nnl2_log_current_config.enable_timestamps = enabled;
}

void nnl2_log_set_debug_info(int enabled) {
    nnl2_log_current_config.enable_debug_info = enabled;
}

void nnl2_log_set_min_level(nnl2_log_level_t min_level) {
    if (min_level >= NNL2_LOG_LEVEL_DEBUG && min_level <= NNL2_LOG_LEVEL_FATAL) {
        nnl2_log_current_config.min_log_level = min_level;
    }
}

void nnl2_log_get_config(nnl2_log_config_t* out_config) {
    if (out_config != NULL) {
        *out_config = nnl2_log_current_config;
    }
}	

// Main logging function
void nnl2_log(nnl2_log_level_t level, const char* file, int line, const char* func, const char* format, ...) {
    // Checking the logging level
    if (level < nnl2_log_current_config.min_log_level) {
        return;
    }

    static const char* level_strings[] = {
        NNL2_LOG_PREFIX_DEBUG,
        NNL2_LOG_PREFIX_INFO,
        NNL2_LOG_PREFIX_WARNING,
        NNL2_LOG_PREFIX_ERROR,
        NNL2_LOG_PREFIX_FATAL
    };
    
    static const char* level_colors[] = {
        NNL2_COLOR_DEBUG,
        NNL2_COLOR_INFO,
        NNL2_COLOR_WARNING,
        NNL2_COLOR_ERROR,
        NNL2_COLOR_FATAL
    };

    char timestamp[NNL2_LOG_TIMESTAMP_SIZE] = {0};
    if (nnl2_log_current_config.enable_timestamps) {
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(timestamp, sizeof(timestamp), NNL2_LOG_TIMESTAMP_FORMAT, tm_info);
    }

    // Saving a system error
	int saved_errno = errno;
	
    if (level == NNL2_LOG_LEVEL_DEBUG) {
        saved_errno = 0;
    }
	
    const char* sys_error = (saved_errno != 0) ? strerror(saved_errno) : "";

    va_list args;
    va_start(args, format);

    // Prefix output
    if (nnl2_log_current_config.enable_timestamps) {
        if (nnl2_log_current_config.enable_color) {
            fprintf(stderr, "%s[%s] [%s]", level_colors[level], timestamp, level_strings[level]);
        } else {
            fprintf(stderr, "[%s] [%s]", timestamp, level_strings[level]);
        }
    } else {
        if (nnl2_log_current_config.enable_color) {
            fprintf(stderr, "%s[%s]", level_colors[level], level_strings[level]);
        } else {
            fprintf(stderr, "[%s]", level_strings[level]);
        }
    }

    // Debugging info
	if (nnl2_log_current_config.enable_debug_info && file != NULL) {
		fprintf(stderr, " [%s:%d", file, line);
		if (func != NULL) {
			fprintf(stderr, ":%s", func);
		}
		
		fprintf(stderr, "]");
	}

    fprintf(stderr, ": ");

    // Main message
    vfprintf(stderr, format, args);

    // System error
    if (saved_errno != 0) {
        fprintf(stderr, NNL2_LOG_SYSTEM_ERROR_PREFIX "%d" NNL2_LOG_SYSTEM_ERROR_SUFFIX "%s" NNL2_LOG_SYSTEM_ERROR_END, 
                saved_errno, sys_error);
    }

    // Reset color and translate rows
    if (nnl2_log_current_config.enable_color) {
        fprintf(stderr, NNL2_COLOR_RESET);
    }
	
    fprintf(stderr, "\n");

    va_end(args);

    // Restoring errno
    errno = saved_errno;
}

#endif
