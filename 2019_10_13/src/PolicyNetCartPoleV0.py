import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl


class PolicyNetCartPoleV0:

    def __init__(self, batch_size=25, action_space=4, hidden_size=30, total_episodes=1000):
        self.batch_size = batch_size
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.total_episodes = total_episodes
        self.episode_number = 0

        self.env = gym.make("CartPole-v0")

        # 环境
        self.input_observation = tf.placeholder(tf.float32, [None, self.action_space], name="input_x")
        # 虚拟label
        self.input_action = tf.placeholder(tf.float32, [None, 1], name="input_action")
        # 每个Action的潜在价值
        self.advantages = tf.placeholder(tf.float32, name="reward_signal")

        self.w1_grad = tf.placeholder(tf.float32, name="batch_grad1")
        self.w2_grad = tf.placeholder(tf.float32, name="batch_grad2")

        self.probability = None
        self.new_grads = None
        self.update_grads = None

        self.t_vars = None
        self.log_like = None
        self.loss = None
        self.adam = None
        self.sess = None

        # 累计reward
        self.reward_sum = 0
        self.rendering = False
        # environments:环境信息/actions:label列表/action_rewards:Action Reward
        self.environments, self.actions, self.action_rewards = [], [], []
        pass

    def net(self):
        w1 = tf.get_variable("W1", shape=[self.action_space, self.hidden_size], initializer=tcl.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(self.input_observation, w1))
        w2 = tf.get_variable("W2", shape=[self.hidden_size, 1], initializer=tcl.xavier_initializer())

        # 策略网络输出的概率
        self.probability = tf.nn.sigmoid(tf.matmul(layer1, w2))

        # 当action为1时，概率为p;当action为0时，概率为(1-p)
        self.log_like = tf.log((1 - self.input_action) * (1 - self.probability) + self.input_action * self.probability)

        # 将log_like与潜在价值advantages相乘，并取负数作为损失
        # （让概率*价值最大：让获得较多价值的Action概率变大，获得较少价值的Action概率变小）
        # 经过不断地训练，能持续加大获得较多价值的Action的概率，学到一个获得更多潜在价值的策略。
        self.loss = -tf.reduce_mean(self.log_like * self.advantages)

        self.t_vars = tf.trainable_variables()
        # 求解梯度：执行完一次后求解梯度: 对loss求t_vars的偏导，返回sum(dy/dx)
        self.new_grads = tf.gradients(self.loss, self.t_vars)

        self.adam = tf.train.AdamOptimizer(learning_rate=1e-1)
        # 更新梯度：执行完一批后更新梯度
        # self.update_grads = self.adam.apply_gradients(zip([self.w1_grad, self.w2_grad], self.t_vars))
        self.update_grads = self.adam.apply_gradients([(self.w1_grad, w1), (self.w2_grad, w2)])
        pass

    def runner(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 初始化
        observation = self.env.reset()

        # 参数缓冲器：在这里是W1和W2
        grad_buffer = self.sess.run(self.t_vars)
        for index, grad in enumerate(grad_buffer):
            grad_buffer[index] = grad * 0

        while self.episode_number <= self.total_episodes:
            # 控制是否渲染
            if self.reward_sum / self.batch_size > 199 or self.rendering:
                self.env.render()
                self.rendering = True

            """
           通过网络，根据“当前的环境”得到“当前的决策”执行后获取“当前的奖励”和“下一步的环境”
           """

            observation = np.reshape(observation, [1, self.action_space])

            # 得到下一次的Action概率
            tf_prob = self.sess.run(self.probability, feed_dict={self.input_observation: observation})
            action = 1 if np.random.uniform() < tf_prob else 0

            # 存储环境
            self.environments.append(observation)
            # action label
            self.actions.append(action)

            # 执行下一步
            observation, reward, done, info = self.env.step(action)
            # 累计奖励
            self.reward_sum += reward
            # 存放reward
            self.action_rewards.append(reward)

            if done:
                self.episode_number += 1
                # 一次实验中获得的所有的observation、label、reward列表
                ep_environments = np.vstack(self.environments)
                ep_actions = np.vstack(self.actions)
                ep_action_rewards = np.vstack(self.action_rewards)

                # 清空
                self.environments, self.actions, self.action_rewards = [], [], []

                # 折扣后的奖励
                discounted_epr = self.discount_rewards(ep_action_rewards)
                # 标准化
                discounted_epr = discounted_epr - np.mean(discounted_epr)
                discounted_epr = discounted_epr / np.std(discounted_epr)

                # 求解梯度
                t_grad = self.sess.run(self.new_grads,
                                       feed_dict={self.input_observation: ep_environments,
                                                  self.input_action: ep_actions,
                                                  self.advantages: discounted_epr})

                # 更新梯度
                for index, grad in enumerate(t_grad):
                    grad_buffer[index] += grad
                    pass

                # 积累一批的数据
                if self.episode_number % self.batch_size == 0:
                    # 更新梯度
                    self.sess.run(self.update_grads,
                                  feed_dict={self.w1_grad: grad_buffer[0], self.w2_grad: grad_buffer[1]})

                    # 初始化梯度
                    for index, grad in enumerate(grad_buffer):
                        grad_buffer[index] = grad * 0

                    print("Average reward {}：{}".format(self.episode_number, self.reward_sum / self.batch_size))

                    if self.reward_sum / self.batch_size > 200:
                        print("Task solved in {} episodes!".format(self.episode_number))
                        break
                    self.reward_sum = 0

                # 重置环境
                observation = self.env.reset()

        pass

    @staticmethod
    def discount_rewards(r, gamma=0.99):
        """
        奖励折扣：Action越靠前，其潜在价值越大
        :param r: 
        :param gamma: 
        :return: 
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    pass


if __name__ == '__main__':
    policy_net = PolicyNetCartPoleV0()
    policy_net.net()
    policy_net.runner()
