import numpy as np
import itertools
import scipy.misc
import random
import os
import time
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.layers as tcl


class Tool:

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    pass


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

    def __init__(self, size=5, scalar=17):
        # 物体的大小
        self.scalar = scalar
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
        b = scipy.misc.imresize(a[:, :, 0], [self.size_x * self.scalar, self.size_x * self.scalar, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [self.size_x * self.scalar, self.size_x * self.scalar, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [self.size_x * self.scalar, self.size_x * self.scalar, 1], interp='nearest')
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

    def __init__(self, size=5):
        environment = GameEnvironment(size)

        state, reward, done = environment.step(1)
        print("reward={} done={}".format(reward, done))
        environment.show_environment()

        state, reward, done = environment.step(1)
        print("reward={} done={}".format(reward, done))
        environment.show_environment()
        pass

    pass


# DQN
class QNetwork:

    def __init__(self, environment, input_size, channel):
        self.channel = channel
        # 环境的大小：85
        self.input_size = input_size
        conv_final_size = 512
        # 输入
        self.scalar_input = tf.placeholder(shape=[None, self.input_size * self.input_size * self.channel],
                                           dtype=tf.float32)
        # 变形为图片输入
        self.image_in = tf.reshape(self.scalar_input, shape=[-1, self.input_size, self.input_size, self.channel])

        # 85:20*20*32
        self.conv1 = tcl.convolution2d(inputs=self.image_in, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                       padding='VALID', biases_initializer=None)
        # 9*9*64
        self.conv2 = tcl.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                                       padding='VALID', biases_initializer=None)
        # 7*7*64
        self.conv3 = tcl.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
                                       padding='VALID', biases_initializer=None)
        # 1*1*512
        self.conv4 = tcl.convolution2d(inputs=self.conv3, num_outputs=conv_final_size, kernel_size=[7, 7],
                                       stride=[1, 1], padding='VALID', biases_initializer=None)

        # 拆成两份：Action的价值，环境本身的价值
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        # 拉直
        self.streamA = tcl.flatten(self.streamAC)
        self.streamV = tcl.flatten(self.streamVC)

        self.AW = tf.Variable(tf.random_normal([conv_final_size // 2, environment.actions]))
        self.VW = tf.Variable(tf.random_normal([conv_final_size // 2, 1]))
        # Action的价值
        self.Advantage = tf.matmul(self.streamA, self.AW)
        # 环境的价值
        self.Value = tf.matmul(self.streamV, self.VW)

        # Value + Advantage
        self.q_out = self.Value + tf.subtract(self.Advantage,
                                              tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        # Q值最大的Action
        self.predict = tf.argmax(self.q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, environment.actions, dtype=tf.float32)

        # 预测的Q值
        self.Q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_one_hot), reduction_indices=1)

        # loss
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        # 训练
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        pass

    pass


# 经验buffer
class ExperienceBuffer:

    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
        pass

    def add(self, experience):
        # 若经验超过最大容量，清除最前面的经验
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(experience) + len(self.buffer) - self.buffer_size] = []
        # add
        self.buffer.extend(experience)
        pass

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    pass


class Runner:

    def __init__(self, environment, environment_size=5, environment_scalar=17, channel=3):
        self.environment = environment
        self.environment_size = environment_size
        self.environment_scalar = environment_scalar
        self.input_size = environment_size * environment_scalar
        self.channel = channel

        self.batch_size = 32
        # 更新频率
        self.update_freq = 4
        # dicount factor
        self.discount_factor = 0.99
        # 起始执行随机Action的概率
        self.start_e = 1.0
        # 最终执行随机Action的概率
        self.end_e = 0.1
        # 从初始随机概率降到最终随机概率的步数
        self.anneling_steps = 10000.0
        # step drop
        self.step_drop = (self.start_e - self.end_e) / self.anneling_steps
        # 进行试验的次数
        self.num_episodes = 10000
        # 预训练步数
        self.pre_train_steps = 10000
        # 每次试验进行的Action步数
        self.max_episode_length = 50
        # 是否读取保存的模型
        self.is_load_model = False
        # target DQN 向主DQN学习的速率
        self.tau = 0.001
        # 模型保存的路径
        self.path = Tool.new_dir("../model/dqn")

        # train about
        self.sess = None
        self.saver = None
        self.target_ops = None
        self.my_buffer = None

        # 奖励列表
        self.reward_list = []
        # 总步长
        self.total_steps = 0
        pass

    def main(self):
        tf.reset_default_graph()

        main_dqn = QNetwork(self.environment, self.input_size, self.channel)
        target_dqn = QNetwork(self.environment, self.input_size, self.channel)

        self.target_ops = self.update_target_graph(tf.trainable_variables(), self.tau)
        self.my_buffer = ExperienceBuffer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        # 加载模型
        if self.is_load_model:
            Tool.print_info("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(self.path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            pass

        # 初始化
        self.sess.run(tf.global_variables_initializer())

        # 更新
        self.update_target(self.target_ops, self.sess)

        # 每一代
        for i in range(self.num_episodes + 1):
            episode_buffer = ExperienceBuffer()
            # 重置环境，得到初始状态
            state = self.process_state(self.environment.reset())

            prob = self.start_e
            reward_all = 0
            step = 0
            while step < self.max_episode_length:
                step += 1
                if np.random.rand(1) < prob or self.total_steps < self.pre_train_steps:
                    # 起始Action或预训练
                    action = np.random.randint(0, 4)
                else:
                    # 预测
                    action = self.sess.run(main_dqn.predict, feed_dict={main_dqn.scalar_input: [state]})[0]

                # 执行Action
                now_state, reward, done = self.environment.step(action)
                now_state = self.process_state(now_state)

                # 增加步数
                self.total_steps += 1

                # 储存经验
                episode_buffer.add(np.reshape(np.array([state, action, reward, now_state, done]), [1, 5]))

                if self.total_steps > self.pre_train_steps:
                    # 概率衰减
                    if prob > self.end_e:
                        prob -= self.step_drop

                    if self.total_steps % self.update_freq == 0:
                        train_batch = self.my_buffer.sample(self.batch_size)
                        predict = self.sess.run(main_dqn.predict,
                                                feed_dict={main_dqn.scalar_input: np.vstack(train_batch[:, 3])})
                        q_out = self.sess.run(target_dqn.q_out,
                                              feed_dict={target_dqn.scalar_input: np.vstack(train_batch[:, 3])})
                        q_double = q_out[range(self.batch_size), predict]
                        q_target = train_batch[:, 2] + self.discount_factor * q_double
                        _ = self.sess.run(main_dqn.updateModel, feed_dict={
                            main_dqn.scalar_input: np.vstack(train_batch[:, 0]),
                            main_dqn.targetQ: q_target,
                            main_dqn.actions: train_batch[:, 1]
                        })
                        # 更新
                        self.update_target(self.target_ops, self.sess)

                        pass
                    pass

                reward_all += reward
                state = now_state

                if done:
                    break

                pass
            self.my_buffer.add(episode_buffer.buffer)
            self.reward_list.append(reward_all)

            if i > 0 and i % 25 == 0:
                Tool.print_info(
                    "episode {} average reward:{}".format(i, np.mean(self.reward_list[-25:])))
            if i > 0 and i % 1000 == 0:
                self.saver.save(self.sess, self.path + "/model-{}.ckpt".format(i))
                Tool.print_info("Saves Model")
            pass

        # 保存模型
        self.saver.save(self.sess, self.path + "/model-{}.ckpt".format(self.num_episodes))
        pass

    @staticmethod
    def update_target_graph(tf_vars, tau):
        total_vars = len(tf_vars)
        op_holder = []
        for idx, var in enumerate(tf_vars[0: total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars // 2].assign(var.value() * tau +
                                                                   (1 - tau) * tf_vars[idx + total_vars // 2].value()))
        return op_holder

    @staticmethod
    def update_target(op_holder, sess):
        for op in op_holder:
            sess.run(op)
        pass

    def process_state(self, states):
        return np.reshape(states, [self.input_size * self.input_size * self.channel])

    pass

if __name__ == '__main__':

    env = GameEnvironment(5)
    runner = Runner(env)
    runner.main()

    result = np.average(np.resize(np.array(runner.reward_list), [len(runner.reward_list) // 100, 100]), 1)
    Tool.print_info(result)

    pass
