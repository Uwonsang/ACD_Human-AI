from . import VecEnvWrapper
from baselines.bench.monitor import ResultsWriter
import numpy as np
import time
from collections import deque

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0):
        VecEnvWrapper.__init__(self, venv)

        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart})
        else:
            self.results_writer = None
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = []
        for (i, (done, ret, eplen, info)) in enumerate(zip(dones, self.eprets, self.eplens, infos)):
            info = info.copy()
            if done:
                if 'episode' in info.keys():
                    epinfo = info['episode'].update({'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)})
                else:
                    epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
            newinfos.append(info)

        return obs, rews, dones, newinfos