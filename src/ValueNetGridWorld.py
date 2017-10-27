import numpy as np
import itertools
import scipy.misc
from PIL import Image


# 游戏状态
class GameObservation:

    def __init__(self, coordinates, size, intensity, channel, reward, name):
        # x y 坐标
        self.x = coordinates[0]
        self.y = coordinates[1]
        # 尺寸
        self.size = size
        # 亮度值
        self.intensity = intensity
        # 颜色通道
        self.channel = channel
        # 奖励值
        self.reward = reward
        # 名称
        self.name = name
        pass

    pass


# 游戏环境
class GameEnvironment:

    def __init__(self, size):
        # 环境的大小
        self.size_x = size
        self.size_y = size
        # 环境的Action Space
        self.actions = 4
        # 环境的物体对象列表
        self.objects = []
        # 重置环境
        self.state = self.reset()
        # 显示最初的环境图像
        self.show_environment()
        pass

    # 重置
    def reset(self):
        # 所有物体的size和intensity为1
        # hero的channel为2，goal的channel为1，fire的channel为0
        self.objects = []

        # 用户控制的对象
        self.objects.append(self.get_hero())

        # 4 个目标对象：奖励+1
        self.objects.append(self.get_goal())
        self.objects.append(self.get_goal())
        self.objects.append(self.get_goal())
        self.objects.append(self.get_goal())

        # 2个fire对象：奖励-1
        self.objects.append(self.get_fire())
        self.objects.append(self.get_fire())

        # 绘制图像
        self.state = self.render_environment()
        return self.state
        pass

    def get_hero(self):
        return GameObservation(self.new_position(), 1, 1, 2, None, 'hero')

    def get_goal(self):
        return GameObservation(self.new_position(), 1, 1, 1, 1, 'goal')

    def get_fire(self):
        return GameObservation(self.new_position(), 1, 1, 0, -1, 'fire')

    # 移动一步
    def move_char(self, direction):
        hero = self.objects[0]
        # 0 - up
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        # 1 - down
        if direction == 1 and hero.y <= self.size_y - 2:
            hero.y += 1
        # 2 - left
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        # 3 - right
        if direction == 3 and hero.x <= self.size_x - 2:
            hero.x += 1
        self.objects[0] = hero
        pass

    # 选择一个跟现有位置不冲突的位置
    def new_position(self):
        position_range = [range(self.size_x), range(self.size_y)]
        # 构造所有的点
        points = []
        for t in itertools.product(*position_range):
            points.append(t)
        # 目前被占用的位置
        current_positions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in current_positions:
                current_positions.append((objectA.x, objectA.y))
        # 从所有的点钟移除被占用的点
        for pos in current_positions:
            points.remove(pos)
        # 随机选择一个位置
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    # 检测是否碰撞
    def check_goal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            # 重叠则说明碰撞了
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(self.get_goal())
                else:
                    self.objects.append(self.get_fire())
                # 返回奖励
                return other.reward, False
        # 没有奖励
        return 0.0, False

    # 渲染环境：绘制图像
    def render_environment(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])
        # 内部赋值为0：黑色
        a[1:-1, 1:-1, :] = 0

        # 绘制物体
        for item in self.objects:
            a[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity

        # resize 图像
        b = scipy.misc.imresize(a[:, :, 0], [self.size_x * 40, self.size_x * 40, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [self.size_x * 40, self.size_x * 40, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [self.size_x * 40, self.size_x * 40, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)

        return a

    def step(self, action):
        # 移动一步
        self.move_char(action)
        # 检查是否触碰
        reward, done = self.check_goal()
        # 过去图像的状态
        self.state = self.render_environment()
        return self.state, reward, done

    # 显示环境
    def show_environment(self):
        Image.fromarray(self.state).convert("RGB").show()
        pass

    pass


# 玩游戏
class GamePlay:

    def __init__(self, size=10):
        env = GameEnvironment(size)

        state, reward, done = env.step(1)
        print("reward={} done={}".format(reward, done))
        env.show_environment()

        state, reward, done = env.step(1)
        print("reward={} done={}".format(reward, done))
        env.show_environment()
        pass

    pass

if __name__ == '__main__':

    GamePlay()

    pass
