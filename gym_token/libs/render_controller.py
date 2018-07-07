import pyqtgraph as pg
import numpy as np

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

red = pg.mkBrush('#ff0000')
blue = pg.mkBrush('#0000ff')
white = pg.mkBrush('#ffffff')


class Render_Controller:
	def __init__(self):
		self.pw = pg.plot()
		self.values = []
		self.actions = []
		print("Create window", type(self.pw))


	def render(self, new_value, new_action=1):
		

		self.values.append(new_value)
		self.actions.append(new_action)

		brushes = []

		for i,u in enumerate(self.values):
			if self.actions[i]==0:
				brushes.append(red)
			elif self.actions[i]==2:
				brushes.append(blue)
			else:
				brushes.append(white)

		self.pw.plot(self.values, symbolBrush=brushes, symbol='o', clear=True)
		pg.QtGui.QApplication.processEvents()

	def close_window(self):
		# pg.exit()
		pass

if __name__=="__main__":
	rc= Render_Controller()
	while True:
		rc.render(np.random.uniform(), np.random.choice([0,1,2]))
