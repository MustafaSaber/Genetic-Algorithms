import matplotlib.pyplot as plt


class Graph():
    """Draw interactive graph"""

    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        plt.axis([-2, 10, -2, 2])
        self.ax = self.fig.add_subplot(111)
        self.pred_min_line, = self.ax.plot(
            [], [], linestyle='-.', color='green')
        self.org_line, = self.ax.plot([], [], '-b')
        self.example_line, = self.ax.plot([], [], linestyle='-.', color='cyan')
        self.wait_time = 0.000001

    def update_org(self, x_axis, y_axis):
        """Update the original line"""
        self.org_line.set_xdata(x_axis)
        self.org_line.set_ydata(y_axis)
        self.fig.canvas.draw()
        plt.pause(self.wait_time)

    def update_pred(self, x_axis, y_axis):
        """Update the prediction line"""
        self.pred_min_line.set_xdata(x_axis)
        self.pred_min_line.set_ydata(y_axis)
        self.fig.canvas.draw()
        plt.pause(self.wait_time)

    def update_example(self, x_axis, y_axis):
        """Update the line of each generation"""
        self.example_line.set_xdata(x_axis)
        self.example_line.set_ydata(y_axis)
        self.fig.canvas.draw()
        plt.pause(self.wait_time)
