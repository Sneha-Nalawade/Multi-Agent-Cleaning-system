import random
import math
import numpy as np
import time
from collections import defaultdict
import itertools
import heapq

# Load data
data = np.load("envMap_cvatPolygon_parking 2.npy")
# data[63][0] = 1
# data[0][27] = 1

# Define the Node class
class Node:
    def __init__(self, position, g, h):
        self.position = position
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost from current node to goal
        self.f = g + h  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # Manhattan distance as the heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(matrix, node):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for direction in directions:
        neighbor_position = (node.position[0] + direction[0], node.position[1] + direction[1])
        if 0 <= neighbor_position[0] < len(matrix) and 0 <= neighbor_position[1] < len(matrix[0]):
            if matrix[neighbor_position[0]][neighbor_position[1]] != 2:
                neighbors.append(neighbor_position)
    return neighbors

def a_star(matrix, start, goal):
    open_list = []
    heapq.heappush(open_list, Node(start, 0, heuristic(start, goal)))
    g_score = {start: 0}

    while open_list:
        current_node = heapq.heappop(open_list)
        current_position = current_node.position

        if current_position == goal:
            return current_node.g

        neighbors = get_neighbors(matrix, current_node)
        for neighbor_position in neighbors:
            tentative_g_score = g_score[current_position] + 1

            if neighbor_position not in g_score or tentative_g_score < g_score[neighbor_position]:
                g_score[neighbor_position] = tentative_g_score
                heapq.heappush(open_list, Node(neighbor_position, tentative_g_score, heuristic(neighbor_position, goal)))

    return float('inf')

# Define Point class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def __repr__(self):
        return f"({self.x}, {self.y})"

# Define Robot class
class Robot:
    def __init__(self, robo_id, speed, dump_capacity, T0):
        self.robo_id = robo_id
        self.speed = speed
        self.dump_capacity = dump_capacity
        self.current_dump = 0
        self.T0 = T0
        self.time_remaining = float('inf')
        self.last_loc = Point(0, 0)
        self.last_time = 0
        self.tasks = []

    def can_take_task(self, task_loc, task_litter):
        travel_time = self.last_loc.distance_to(task_loc) / self.speed
        total_time = travel_time + task_litter * t
        if self.current_dump + task_litter > self.dump_capacity:
            self.current_dump = 0
            return False, Dumping_area
        elif self.time_remaining - total_time < self.T0:
            return False, Charging_Point
        return True, total_time

    def assign_task(self, task_loc, task_litter, current_time):
        travel_time = self.last_loc.distance_to(task_loc) / self.speed
        total_time = travel_time + task_litter * t
        self.tasks.append(task_loc)
        self.last_loc = task_loc
        self.last_time = current_time + total_time
        self.current_dump += task_litter
        self.time_remaining -= total_time

    def assign1(self, task_loc, task_litter, current_time):
        self.assign_task(task_loc, task_litter, current_time)

class SACostOptimization:
    def __init__(self, rooms_to_be_allocated, centroid, robots, T0=500, r=0.997, Ts=1, Lk=100, maxIterate=10000):
        self.rooms_to_be_allocated = rooms_to_be_allocated
        self.centroid = centroid
        self.robots = robots
        self.T0 = T0  # Initial temperature
        self.r = r  # Cooling rate
        self.Ts = Ts  # Stopping temperature
        self.Lk = Lk  # Number of iterations at each temperature
        self.maxIterate = maxIterate  # Maximum number of iterations

    def initial_job_allocation(self):
        for room in self.rooms_to_be_allocated:
            task_loc = Point(*self.centroid[room])
            task_litter = littering[room]
            min_time = float('inf')
            nearest_robo = None

            for robot in self.robots:
                if robot.can_take_task(task_loc, task_litter)[0]:  # Use can_take_task to check feasibility
                    task_time = robot.last_time
                    if task_time < min_time:
                        min_time = task_time
                        nearest_robo = robot

            if nearest_robo:
                nearest_robo.assign_task(task_loc, task_litter, current_time=nearest_robo.last_time)
        return [(robot.robo_id, robot.tasks) for robot in self.robots]

    def job_allocation(self, task_order=None):
        if task_order is None:
            return self.initial_job_allocation()
        
        for robot in self.robots:
            robot.tasks = []
            robot.current_dump = 0
            robot.last_loc = Point(0, 0)
            robot.time_remaining = float('inf')
            robot.last_time = 0

        for i in task_order:
            task_loc = Point(*self.centroid[self.rooms_to_be_allocated[i]])
            task_litter = littering[self.rooms_to_be_allocated[i]]
            
            assigned = False
            for robot in self.robots:
                can_take, extra_task = robot.can_take_task(task_loc, task_litter)
                if can_take:
                    robot.assign_task(task_loc, task_litter, current_time=robot.last_time)
                    assigned = True
                    break
            
            if not assigned:
                robot.assign_task(extra_task, 0, current_time=robot.last_time)
                robot.assign_task(task_loc, task_litter, current_time=robot.last_time)
        
        return [(robot.robo_id, robot.tasks) for robot in self.robots]

    def calculate_cost(self):
        max_cost = 0
        for robot in self.robots:
            cost = 0
            previous_task = Point(0, 0)
            for task in robot.tasks:
                if isinstance(task, Point):
                    travel_time = previous_task.distance_to(task) / robot.speed
                    cleaning_time = littering.get((task.x, task.y), 0) * t
                    cost += travel_time + cleaning_time
                    previous_task = task
                elif task == Dumping_area:
                    travel_time = previous_task.distance_to(Dumping_area) / robot.speed
                    cost += travel_time
                    previous_task = Dumping_area
                elif task == Charging_Point:
                    travel_time = previous_task.distance_to(Charging_Point) / robot.speed
                    cost += travel_time
                    previous_task = Charging_Point
            max_cost = max(max_cost, cost)
        return max_cost

    def create_neighbor(self, task_order, mode):
        while True:
            if mode == 1:
                new_order = self.swap(task_order)
            elif mode == 2:
                new_order = self.reversion(task_order)
            else:
                new_order = self.swap(task_order)
            if self.is_feasible(new_order):
                break
        return new_order

    def swap(self, task_order):
        if len(task_order) < 2:
            return task_order
        new_order = list(task_order)
        r = random.sample(range(len(task_order)), 2)
        new_order[r[0]], new_order[r[1]] = new_order[r[1]], new_order[r[0]]
        return new_order

    def reversion(self, task_order):
        if len(task_order) < 2:
            return task_order
        new_order = list(task_order)
        r = random.sample(range(len(task_order)), 2)
        i1 = min(r)
        i2 = max(r)
        new_order[i1:i2 + 1] = new_order[i1:i2 + 1][::-1]
        return new_order

    def is_feasible(self, task_order):
        return True

    def main_run(self):
        initial_order = list(range(len(self.rooms_to_be_allocated)))
        Solu = self.job_allocation()
        Cost = self.calculate_cost()
        T = self.T0

        cnt = 1
        minCost = Cost
        minSolution = Solu
        costArray = np.zeros(self.maxIterate)
        start_time = time.time()

        while T > self.Ts and cnt < self.maxIterate:
            for k in range(self.Lk):
                mode = random.randint(1, 3)
                new_order = self.create_neighbor(initial_order, mode)
                newSolu = self.job_allocation(new_order)
                newCost = self.calculate_cost()
                delta = newCost - Cost
                if delta < 0:
                    Cost = newCost
                    Solu = newSolu
                    initial_order = new_order
                else:
                    p = math.exp(-delta / T)
                    if random.random() <= p:
                        Cost = newCost
                        Solu = newSolu
                        initial_order = new_order

            costArray[cnt] = Cost
            if Cost < minCost:
                minCost = Cost
                minSolution = Solu
            T = T * self.r
            cnt += 1

        end_time = time.time()
        return minSolution, minCost, end_time - start_time

def find_rooms_with_ones(matrix, n):
    M, N = matrix.shape
    rooms_to_be_allocated = []
    centroid = {}
    littering = {}

    for start_row in range(0, M - n + 1, n):
        for start_col in range(0, N - n + 1, n):
            cell = matrix[start_row:start_row + n, start_col:start_col + n]
            ones_count = np.sum(cell == 1)
            if ones_count > 0:
                centroid_coords = (start_row + n // 2, start_col + n // 2)
                room_id = (start_row // n, start_col // n)
                rooms_to_be_allocated.append(room_id)
                centroid[room_id] = centroid_coords
                littering[room_id] = ones_count

    return rooms_to_be_allocated, centroid, littering

# Example usage
Dumping_area = Point(0, 0)
Charging_Point = Point(0, 10)
speed = 10
num_robots = 3
T0 = 15
dump_capacity = 1000
t = 1 # Time required to clean a single litter

rooms_to_be_allocated, centroid, littering = find_rooms_with_ones(data, 18)
robots = [Robot(i, speed, dump_capacity, T0) for i in range(num_robots)]

sa_optimizer = SACostOptimization(rooms_to_be_allocated, centroid, robots)
minSolution, minCost, elapsed_time = sa_optimizer.main_run()

print("Minimum Solution:", minSolution)
print("Minimum Cost:", minCost)
print("Elapsed Time:", elapsed_time)


