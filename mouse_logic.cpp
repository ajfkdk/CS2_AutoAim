#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <shared_mutex>
#include "mouse_logic.h"



MouseLogic::MouseLogic(int capture_size, int screen_width, int screen_height, int image_top_left_x, int image_top_left_y)
    : capture_size(capture_size), screen_width(screen_width), screen_height(screen_height),
    image_top_left_x(image_top_left_x), image_top_left_y(image_top_left_y), start_x(0), start_y(0), end_x(0), end_y(0), is_head(false), reached_target(true) {}

void MouseLogic::set_boxes(const std::vector<DL_RESULT>& boxes) {
    this->boxes = boxes;
    find_nearest_box();
    calculate_line_segment();
    //只有boxes不为空时才会更新
    if (boxes.size() != 0) {
		reached_target = false;
	}
	
}


void MouseLogic::find_nearest_box() {
    int target_x = capture_size / 2;
    int target_y = capture_size / 2;
    int min_distance_sq = std::numeric_limits<int>::max();

    for (const auto& box : boxes) {
        int bbox_x = (box.box.x + box.box.x + box.box.width) / 2;
        int bbox_y = (box.box.y + box.box.y + box.box.height) / 2;
        int distance_sq = (bbox_x - target_x) * (bbox_x - target_x) + (bbox_y - target_y) * (bbox_y - target_y);

        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            end_x = bbox_x + image_top_left_x;
            end_y = bbox_y + image_top_left_y;
            is_head = (box.classId == 1 || box.classId == 3);
        }
    }
}

void MouseLogic::calculate_line_segment() {
    start_x = screen_width / 2;
    start_y = screen_height / 2;
}

//重置end_x和end_y
void MouseLogic::reset_end() {
    start_x = screen_width / 2;
    start_y = screen_height / 2;
	end_x = start_x;
	end_y = start_y;
}


std::pair<int, int> MouseLogic::get_move_vector(float speed) {
    if (reached_target) {
        return { 0, 0 }; // 已到达目标点，不需要继续移动
    }

    int dx = end_x - start_x;
    int dy = end_y - start_y;

    int max_step = 10;
    int min_step = 1;

    int move_x = 0;
    if (dx != 0) {
        int distance_x = std::abs(dx);

        float step_x = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance_x / 10.0f));
        step_x *= speed;
        move_x = static_cast<int>(dx / distance_x * step_x);
       /* std::cout <<"----------------------"<<std::endl;
        std::cout <<"start_x:"<<start_x<<std::endl;
        std::cout <<"end_x:"<<end_x<<std::endl;
       std::cout<<"move_x:"<<move_x<<std::endl;
       std::cout <<"dx:"<<dx<<std::endl;
       std::cout <<"distance_x:"<<distance_x<<std::endl;*/

    }


    int move_y = 0;
    if (is_head && dy != 0) {
        int distance_y = std::abs(dy);
        float step_y = std::max(static_cast<float>(min_step), std::min(static_cast<float>(max_step), distance_y / 10.0f));
        step_y *= speed;
        move_y = static_cast<int>(dy / distance_y * step_y);
    }

    // 计算移动矢量的长度平方
    int move_distance_sq = move_x * move_x + move_y * move_y;
    // 计算剩余距离的平方
    int remaining_distance_sq = dx * dx + dy * dy;

    if (move_distance_sq >= remaining_distance_sq) {
        // 如果移动矢量长度大于或等于剩余距离长度，则返回剩余的长度，并标记已到达目标点
        move_x = dx;
        move_y = dy;
        reached_target = true;
        end_x = start_x;
        end_y = start_y;
    }
    else {
        // 更新线段起点
        start_x += move_x;
        start_y += move_y;

    }
    /*std::cout <<"move_distance_sq:"<<move_distance_sq<<std::endl;
    std::cout <<"remaining_distance_sq:"<<remaining_distance_sq<<std::endl;*/
    return { move_x, move_y };
}