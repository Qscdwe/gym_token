import math
import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
import pandas as pd
import datetime
import pytz

import pkg_resources


class TokenEnv(gym.Env):
	metadata = {
		'render.modes': ['human']
	}

	def __init__(self):
		# open price, high price, low price, close price, day in week, hour in day, vol, quote_asset_vol, base_asset_vol, trades, base_wallet, quote_wallet
		# n = 12
		low_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
		high_space = np.array([1,1,1,1,7,24,5000,5000,5000,10000,1000,1000])

		resource_package = 'gym_token'
		resource_path = '/'.join(('envs', 'DLTETH.csv'))
		csv_file = pkg_resources.resource_filename(resource_package, resource_path)
	
		self.observation_space = spaces.Box(low_space, high_space)		
		self.action_space = spaces.Discrete(3)
		self.tick = 0
		self.base_wallet = 0
		self.quote_wallet = 0
		self.last_quote_wallet = self.quote_wallet


		print(f'Reading file {csv_file}')
		self.file = open(csv_file)
		self.data = pd.read_csv(csv_file)

	def get_state(self, tick):
		def get_day(unix_timestamp):
			date = datetime.datetime.fromtimestamp(int(unix_timestamp))
			utc_dt = date.astimezone(pytz.utc)
			return utc_dt.weekday()
		def get_hour(unix_timestamp):
			date = datetime.datetime.fromtimestamp(int(unix_timestamp))
			utc_dt = date.astimezone(pytz.utc)
			return utc_dt.hour

		day_in_week = get_day(self.data['openTime'][tick]//1000)
		hour_in_day = get_hour(self.data['openTime'][tick]//1000)

		state = [ float(self.data['open'][tick]), 
						float(self.data['high'][tick]), 
						float(self.data['low'][tick]), 
						float(self.data['close'][tick]), 
						float(day_in_week), 
						float(hour_in_day),
						float(self.data['volume'][tick]),
						float(self.data['quoteAssetVolume'][tick]),
						float(self.data['baseAssetVolume'][tick]), 
						float(self.data['trades'][tick]), 
						float(self.base_wallet), 
						float(self.quote_wallet)]
		state = np.array(state).reshape(12,)
		return state

	def reset(self):
		self.debug_total_return_reward = 0
		self.debug_action_sequence = []
		self.tick = 0
		# Train with ~30 initial capital money
		base_wallet = 1 + np.random.uniform()*29
		quote_wallet = 0
		self.base_wallet = base_wallet
		self.quote_wallet = quote_wallet
		self.last_quote_wallet = self.quote_wallet
		self.state = self.get_state(self.tick)

		print('TokenEnv loaded.')
		print(f'Initialized with {base_wallet} Bcoin and {quote_wallet} Qcoin')

		return self.state

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

		self.debug_action_sequence.append(action)
		# current state at asked tick
		s = self.get_state(self.tick)
		current_price = s[3]
		if action==0:
			# Sell action
			# Ban mot luong da duoc floor. Vd: 13DLT
			costed_base_wallet = np.floor(self.base_wallet)
			tmp_wallet = costed_base_wallet * current_price * 99.9/100
			self.quote_wallet += tmp_wallet
			self.base_wallet -= costed_base_wallet
			reward = self.quote_wallet - self.last_quote_wallet
			self.last_quote_wallet = self.quote_wallet

		elif action==1:
			#Hold action
			reward = 0
		elif action==2:
			# Buy action
			# Mua mot luong floor. vd: 7DLT.
			tmp_wallet = self.quote_wallet * (1/current_price) * (99.9/100)
			tmp_wallet = np.floor(tmp_wallet)
			costed_quote_wallet = tmp_wallet / (99.9/100) / (1/current_price)
			self.base_wallet += tmp_wallet
			self.quote_wallet -= costed_quote_wallet
			reward = self.quote_wallet - self.last_quote_wallet
			self.last_quote_wallet = self.quote_wallet

		done = False
		if self.tick >=3998:
			done = True
			print('DEBUG ACTION: ', self.debug_action_sequence[-10:])
			print('DEBUG REWARD: ', self.debug_total_return_reward)
			print('==== end debug ====')
			# print(f'ACcess {done}')

		# print(f'tick {self.tick}, {done}')

		# Update state according to new tick and new wallet
		# Ignore the last line of csv file
		self.tick+=1
		s_ = self.get_state(self.tick)

		# Magnify reward *= 1
		self.debug_total_return_reward += reward
		return s_, reward, done, {}

	def render(self, mode='human'):
		pass

	def close(self):
		# if self.viewer: self.viewer.close()
		pass
