#include <common/defines.hpp>
#include <fg/exception.h>

#define FG_THROW(fn)                                                         \
    do {                                                                     \
        fg_err __err = fn;                                                   \
        if (__err == FG_ERR_NONE) break;                                     \
        char *msg = NULL;                                                    \
        fg_get_last_error(&msg, NULL);                                       \
        forge::Error ex(msg, __PRETTY_FUNCTION__, __FG_FILENAME__, __LINE__, \
                        __err);                                              \
        delete[] msg;                                                        \
        throw ex;                                                            \
    } while (0)

#define FG_THROW_ERR(__msg, __err)                                      \
    do {                                                                \
        throw forge::Error(__msg, __PRETTY_FUNCTION__, __FG_FILENAME__, \
                           __LINE__, __err);                            \
    } while (0)
