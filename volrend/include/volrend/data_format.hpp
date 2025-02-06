#pragma once

#include "volrend/common.hpp"
#include <string>

namespace volrend {

struct DataFormat {
    enum {
        RGBA,  // Simply stores rgba
        SH,
        SG,
        ASG,
        FC,
        LFC,
        _COUNT,
    } format;

    // SH/SG/ASG dimension per channel
    int basis_dim = -1;

    // FC dimensions for rgb and sigma
    int fc_dim_sigma = -1;
    int fc_dim_rgb = -1;

    // Parse a string like 'SH16', 'SG25'
    void parse(const std::string& str);

    // Convert to string
    std::string to_string() const;
};

}  // namespace volrend
