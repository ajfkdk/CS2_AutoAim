#ifndef MOUSE_LOGIC_H
#define MOUSE_LOGIC_H

#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <shared_mutex>
#include "global_config.h"

class MouseLogic {
public:
    MouseLogic(int capture_size, int screen_width, int screen_height, int image_top_left_x, int image_top_left_y);
    void set_boxes(const std::vector<DL_RESULT>& boxes);
    std::pair<int, int> get_move_vector(float speed = 1.0f);
    void MouseLogic::reset_end();

private:
    int capture_size;
    int screen_width;
    int screen_height;
    int image_top_left_x;
    int image_top_left_y;
    std::vector<DL_RESULT> boxes;

    int start_x, start_y, end_x, end_y;
    bool is_head;
    bool reached_target;

    mutable std::shared_mutex mtx;

    void find_nearest_box();
    void calculate_line_segment();
};

#endif // MOUSE_LOGIC_H