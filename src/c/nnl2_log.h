#ifndef NNL2_LOG_H
#define NNL2_LOG_H

#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>

// NNL2

/** @file nnl2_log.h
 ** @brief Full implementation of the logging system
 ** @copyright MIT License
 ** @defgroup logging Logging System
 *
 * File provides a simple yet powerful logging system
 * with colored terminal output, support for timestamps, and
 * automatic handling of system errors (errno)
 *
 ** Filepath: nnl2/src/c/nnl2_log.h
 ** File: nnl2_log.h
 **
 ** The file contains the implementation of the logging system
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **
 ** @code
 * int main() {
 *     // Initialization of the logging system
 *     nnl2_log_init( 
 *	       NNL2_LOG_DEFAULT_COLOR,
 *	       NNL2_LOG_DEFAULT_TIMESTAMPS,     
 *	       NNL2_LOG_DEFAULT_DEBUG_INFO,
 *	       NNL2_LOG_DEFAULT_MIN_LEVEL // or NNL2_LOG_LEVEL_DEBUG
 *     );
 *    
 *     // You can also write nnl2_init_system() which 
 *     // automatically initializes the logging system
 *	   // (instead of nnl2_log_init(NNL2_ ...) you can 
 *	   //  use nnl2_init_system())
 *
 *     int32_t qux = 123; // Placeholder
 *
 *     NNL2_INFO("Application started");
 *     NNL2_DEBUG("Value of qux: %d", qux); // You need to set the debug mode by passing the -DNNL2_DEBUG_MODE flag
 *	   NNL2_LOG_ERROR("Failed to open file");
 *     
 *     return 0;
 * }
 ** @endcode
 **
 **/

/// @defgroup colors ANSI Colors Constants
/// @{
	
#define NNL2_COLOR_RESET   "\x1b[0m"   ///< Reset all formatting and colors
#define NNL2_COLOR_DEBUG   "\x1b[36m"  ///< Cyan - debug messages
#define NNL2_COLOR_INFO    "\x1b[32m"  ///< Green - informational message
#define NNL2_COLOR_WARNING "\x1b[33m"  ///< Yellow - warning messages
#define NNL2_COLOR_ERROR   "\x1b[31m"  ///< Red - error messages
#define NNL2_COLOR_FATAL   "\x1b[35m"  ///< Magenta - fatal error messages

/// @}

/// @defgroup prefixes Log Prefix Constants
/// @{

#define NNL2_LOG_PREFIX_DEBUG   "DEBUG"   ///< Debug level prefix
#define NNL2_LOG_PREFIX_INFO    "INFO"    ///< Info level prefix
#define NNL2_LOG_PREFIX_WARNING "WARN"    ///< Warning level prefix
#define NNL2_LOG_PREFIX_ERROR   "ERROR"   ///< Error level prefix
#define NNL2_LOG_PREFIX_FATAL   "FATAL"   ///< Fatal level prefix

/// @}

/// @defgroup formatting Formatting Constants  
/// @{

#define NNL2_LOG_TIMESTAMP_FORMAT "%H:%M:%S"  ///< Time format for timestamps
#define NNL2_LOG_TIMESTAMP_SIZE   20          ///< Buffer size for timestamp

#define NNL2_LOG_SYSTEM_ERROR_PREFIX " (system error "  ///< System error prefix
#define NNL2_LOG_SYSTEM_ERROR_SUFFIX ": "               ///< System error separator
#define NNL2_LOG_SYSTEM_ERROR_END    ")"                ///< System error suffix

/// @}

/// @defgroup defaults Default Configuration Values
/// @{

#define NNL2_LOG_DEFAULT_COLOR        true    ///< Enable colors by default
#define NNL2_LOG_DEFAULT_TIMESTAMPS   true    ///< Enable timestamps by default
#define NNL2_LOG_DEFAULT_DEBUG_INFO   false   ///< Disable debug info by default
#define NNL2_LOG_DEFAULT_MIN_LEVEL    NNL2_LOG_LEVEL_INFO  ///< Default min level

/// @}

/**
 * @brief Log level enumeration
 * @ingroup logging
 */
typedef enum {
    NNL2_LOG_LEVEL_DEBUG,    ///< Debug messages - detailed development information
    NNL2_LOG_LEVEL_INFO,     ///< Informational messages - normal operation
    NNL2_LOG_LEVEL_WARNING,  ///< Warning messages - potential issues
    NNL2_LOG_LEVEL_ERROR,    ///< Error messages - operation failures
    NNL2_LOG_LEVEL_FATAL     ///< Fatal messages - critical errors causing shutdown
} nnl2_log_level_t;

/**
 * @brief Logging configuration structure
 * @ingroup logging
 */
typedef struct {
    bool enable_color;               ///< Enable colored output (0/1) (false/true)
    bool enable_timestamps;          ///< Enable timestamp display (0/1) (false/true)
    bool enable_debug_info;          ///< Enable debug information (file/line) (0/1) (false/true)
    nnl2_log_level_t min_log_level; ///< Minimum log level to display
} nnl2_log_config_t;

/** 
 * @brief Global logging configuration
 * @ingroup logging
 * @note This variable should be accessed through configuration functions
 * @see nnl2_log_set_color()
 * @see nnl2_log_set_timestamps()
 * @see nnl2_log_set_debug_info()
 * @see nnl2_log_set_min_level()
 */
extern nnl2_log_config_t nnl2_log_current_config;

// Configuration Functions
void nnl2_log_init(bool enable_color, bool enable_timestamps, bool enable_debug_info, nnl2_log_level_t min_log_level);
void nnl2_log_set_color(bool enabled);
void nnl2_log_set_timestamps(bool enabled);
void nnl2_log_set_debug_info(bool enabled);
void nnl2_log_set_min_level(nnl2_log_level_t min_level);
void nnl2_log_get_config(nnl2_log_config_t* out_config);

// Main logging function
void nnl2_log(nnl2_log_level_t level, const char* file, int line, const char* func, const char* format, ...);

/// @name Logging Macros
/// @brief Convenience macros for different log levels
/// @{
	
/// @brief Generic log macro with auto file/line/function capture	
#define NNL2_LOG(level, ...) nnl2_log(level, __FILE__, __LINE__, __func__, __VA_ARGS__) 
#define NNL2_DEBUG(...)      NNL2_LOG(NNL2_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define NNL2_INFO(...)       NNL2_LOG(NNL2_LOG_LEVEL_INFO, __VA_ARGS__)
#define NNL2_WARN(...)       NNL2_LOG(NNL2_LOG_LEVEL_WARNING, __VA_ARGS__)
#define NNL2_ERROR(...)      NNL2_LOG(NNL2_LOG_LEVEL_ERROR, __VA_ARGS__)
#define NNL2_FATAL(...)      NNL2_LOG(NNL2_LOG_LEVEL_FATAL, __VA_ARGS__)

/** 
 * @brief Log function entry (debug level)
 * @note Automatically captures function name using __func__
 */
#define NNL2_FUNC_ENTER() NNL2_DEBUG("Function %s() entered", __func__)

/** 
 * @brief Log function exit (debug level)
 * @note Automatically captures function name using __func__
 */
#define NNL2_FUNC_EXIT()  NNL2_DEBUG("Function %s() exited", __func__)

/// @}

// --------------- Realization ---------------

// Global configuration
nnl2_log_config_t nnl2_log_current_config = {
    .enable_color = true,
    .enable_timestamps = true,
    .enable_debug_info = false,
    .min_log_level = NNL2_LOG_LEVEL_DEBUG  // By default all levels are
};

// Configuration functions

/// @brief Initialize logging system with custom settings
void nnl2_log_init(bool enable_color, bool enable_timestamps, 
				   bool enable_debug_info, nnl2_log_level_t min_log_level) {
    nnl2_log_current_config.enable_color = enable_color;
    nnl2_log_current_config.enable_timestamps = enable_timestamps;
    nnl2_log_current_config.enable_debug_info = enable_debug_info;
    nnl2_log_current_config.min_log_level = min_log_level;
}

/// @brief Enable/disable colored output
void nnl2_log_set_color(bool enabled) {
    nnl2_log_current_config.enable_color = enabled;
}	

/// @brief Enable/disable timestamp display  
void nnl2_log_set_timestamps(bool enabled) {
    nnl2_log_current_config.enable_timestamps = enabled;
}

/// @brief Enable/disable debug info (file/line)
void nnl2_log_set_debug_info(bool enabled) {
    nnl2_log_current_config.enable_debug_info = enabled;
}

/// @brief Set minimum log level to display
void nnl2_log_set_min_level(nnl2_log_level_t min_level) {
    if (min_level >= NNL2_LOG_LEVEL_DEBUG && min_level <= NNL2_LOG_LEVEL_FATAL) {
        nnl2_log_current_config.min_log_level = min_level;
    }
}

/// @brief Get current logging configuration
void nnl2_log_get_config(nnl2_log_config_t* out_config) {
    if (out_config != NULL) {
        *out_config = nnl2_log_current_config;
    }
}	

/** @brief
 * The main logging function of the nnl2 system
 *
 ** @details 
 * Function performs formatted output of log messages with 
 * support for color formatting via ANSI escape codes, time stamps 
 * with a custom format, information about the source file, line, 
 * and function, automatic handling of system errors (errno), and filtering 
 * by logging levels
 *
 **
 ** @param level
 * Message importance level (from nnl2_log_level_t)
 *
 ** @param file 
 * Name of the source file (usually __FILE__)
 *
 ** @param line
 * Line number in the source file (usually __LINE__)
 *
 ** @param func
 * Function name (usually __func__)
 *
 ** @param format 
 * Printf-style format string
 *
 ** @param ...
 * Variable arguments for formatting
 *
 **
 ** @note
 * The function automatically saves and restores the errno value
 *
 ** @note
 * Messages are filtered according to nnl2_log_current_config.min_log_level
 *
 ** @note
 * For system errors, a description is automatically added from strerror()
 *
 **
 ** @warning
 * Do not call directly - use the macros NNL2_LOG(), NNL2_DEBUG(), etc.
 *
 **
 ** @par Example:
 ** @code
 * // Instead of a direct call:
 * // nnl2_log(NNL2_LOG_LEVEL_ERROR, "file.c", 42, "main", "Error: %s", "message");
 *
 * // Use macros:
 * NNL2_ERROR("Error: %s", "message");
 ** @endcode
 **
 ** @see nnl2_log_config_t
 ** @see nnl2_log_level_t
 ** @see NNL2_LOG()
 ** @see NNL2_DEBUG()
 ** @see NNL2_INFO()
 ** @see NNL2_WARN()
 ** @see NNL2_ERROR()
 ** @see NNL2_FATAL()
 **
 ** @ingroup logging
 **/
void nnl2_log(nnl2_log_level_t level, const char* file, int line, const char* func, const char* format, ...) {
    // Checking the logging level	
    if (level < nnl2_log_current_config.min_log_level) {
        return;
    }
	
	if (format == NULL) {
        fprintf(stderr, "[nnl2] [ERROR]: Null format string passed to nnl2_log\n");
        return;
    }
	
	// Checking the correctness of the logging level
	if (level < NNL2_LOG_LEVEL_DEBUG || level > NNL2_LOG_LEVEL_FATAL) {
        fprintf(stderr, "[nnl2] [ERROR]: Invalid log level: %d\n", level);
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
	
	if (nnl2_log_current_config.enable_color) {
        fprintf(stderr, "%s[nnl2] ", level_colors[level]);
    } else {
        fprintf(stderr, "[nnl2] ");
    }

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
