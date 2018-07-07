import gym
from gym import spaces
import numpy as np
import pandas as pd
import datetime, pytz, pkg_resources, sys

sys.path.insert(0, '../libs')
from ..libs.render_controller import Render_Controller


class TokenEnvFragment(gym.Env):
	def __init__(self):

		# open price, high price, low price, close price, day in week, hour in day, vol, quote_asset_vol, base_asset_vol, trades, base_wallet, quote_wallet
		# n = 12
		low_space = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
		high_space = np.array([1,1,1,1,7,24,5000,5000,5000,10000,1000,1000])

		# Load data
		resource_package = 'gym_token'
		resource_path = '/'.join(('envs', 'DLTETH.csv'))
		self.csv_file = pkg_resources.resource_filename(resource_package, resource_path)
	
		self.observation_space = spaces.Box(low_space, high_space)		
		self.action_space = spaces.Discrete(3)
		self.tick = 0
		self.base_wallet = 0
		self.quote_wallet = 0
		self.last_quote_wallet = self.quote_wallet

		self.file = open(self.csv_file)
		self.data = pd.read_csv(self.csv_file)

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
		self.data_size = len(self.data)

		# A 400-minute trading session
		self.num_step = 400
		self.MAG_REWARD = 100

		#Handle numpy random: 400 - 400 = 0 => error, which we want to happen but numpy doesnt allow (low=high)
		if self.data_size == self.num_step:
			self.start_step = 0
		elif self.data_size > self.num_step:
			self.start_step = np.random.randint(self.data_size - self.num_step)
		else:
			raise ValueError("Input data has size < number of step per episode.")

		self.tick = self.start_step

		# Train with ~ [15,30] initial capital money
		base_wallet = 15 + np.random.uniform()*15
		quote_wallet = 0
		self.base_wallet = base_wallet
		self.quote_wallet = quote_wallet
		self.last_quote_wallet = self.quote_wallet
		self.state = self.get_state(self.tick)

		print('TokenEnvFragment loaded.')
		print(f'Reading file {self.csv_file}')
		print(f'Data size {self.data_size}')
		print(f'Num step will take this episode: {self.num_step}')
		print(f'Range of initializing base wallet: [15, x ,30]')

		
		current_price = self.state[3]
		costed_base_wallet = np.floor(self.base_wallet)
		tmp_wallet = costed_base_wallet * current_price * 99.9/100
		self.init_equivalent_quote_wallet = tmp_wallet

		print(f'Initialized with {base_wallet} Bcoin and {quote_wallet} Qcoin')
		print(f'At initializing time, {base_wallet} Bcoin equivalents {self.init_equivalent_quote_wallet} Qcoin')

		self.rc = None
		self.last_action = 1
		self.price_when_last_action = 0
		return self.state


	def step(self, action):
		def do_action_and_get_reward(action):
			if action==0:
				# Sell BCoin action
				# Sell using a floored amount BCoin. E.g: sell 13BCoin for 0.02 QCoin.
				costed_base_wallet = np.floor(self.base_wallet)
				tmp_wallet = costed_base_wallet * current_price * 99.9/100

				# If action actually do something (like not selling using 0BCoin)
				if tmp_wallet>0:
					self.quote_wallet += tmp_wallet
					self.base_wallet -= costed_base_wallet
					reward = self.quote_wallet - self.last_quote_wallet
					self.last_quote_wallet = self.quote_wallet
				else:
					reward = -0.00001

			elif action==1:
				#Hold action
				reward = 0
			elif action==2:
				# Buy BCoin action
				# Buy a floored amount BCoin. vd: buy 7BCoin using 0.02 QCoin.
				tmp_wallet = self.quote_wallet * (1/current_price) * (99.9/100)
				tmp_wallet = np.floor(tmp_wallet)

				#If actiona actualy do something (like not buying 0BCoin)
				if tmp_wallet>0:
					costed_quote_wallet = tmp_wallet / (99.9/100) / (1/current_price)
					self.base_wallet += tmp_wallet
					self.quote_wallet -= costed_quote_wallet
					reward = self.quote_wallet - self.last_quote_wallet
					self.last_quote_wallet = self.quote_wallet
				else:
					reward = -0.00001
			return reward

		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

		s = self.get_state(self.tick)
		current_price = s[3]

		self.last_action = action
		self.price_when_last_action = current_price
		self.debug_action_sequence.append(action)
		
		reward = do_action_and_get_reward(action)

		done = False
		if self.tick - self.start_step + 1 >= self.num_step:
			done = True
			print('DEBUG ACTION: ', self.debug_action_sequence[-10:])
			print('DEBUG REWARD: ', self.debug_total_return_reward)
			print('DEBUG MAG_REWARD: ', self.MAG_REWARD)
			# Remember there are price fluctuation between the start tick and end tick
			# So this is just a approximate function to estimate gained_Qcoin
			print('DEBUG GAINED_QCOIN:', self.quote_wallet -  self.init_equivalent_quote_wallet)
			print(f'DEBUG BCOIN LEFT: {self.base_wallet}')
			print(f'DEBUG QCOIN LEFT: {self.quote_wallet}')
			print('==== end of episode ====')

		self.tick+=1
		s_ = self.get_state(self.tick)

		self.debug_total_return_reward += reward * self.MAG_REWARD
		return s_, reward * self.MAG_REWARD, done, {}


	def render(self):
		if not self.rc:
			self.rc = Render_Controller()
		self.rc.render(self.price_when_last_action, self.last_action)


	def close(self):
		if self.rc:
			self.rc.close_window()

