#!python3

import tkinter as tk
import json

class guiGraphGrid:
    def __init__(self, fields_x, fields_y, field_size_x, field_size_y, title="Graph", background="white", border_width=0, border_color="black", padding_x=0, padding_y=0):
        self.fields_x = fields_x
        self.fields_y = fields_y
        self.field_size_x = field_size_x
        self.field_size_y = field_size_y
        self.padding_x = padding_x
        self.padding_y = padding_y

        self.window_root = tk.Tk()
        self.window_root.title(title)
        self.canvas = tk.Canvas(self.window_root, width = fields_x*field_size_x, height = fields_y*field_size_y)
        self.canvas.pack()

    def addNode(self, x, y, color="#cccccc", name="", text_size=12):
        x1 = int(x * self.field_size_x + self.padding_x)
        y1 = int(y * self.field_size_y + self.padding_y)
        x2 = int((x+1) * self.field_size_x - self.padding_x)
        y2 = int((y+1) * self.field_size_y - self.padding_y)
        radius = int(self.field_size_x * 0.1)
        
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1,
        ]
        self.canvas.create_polygon(points, fill=color, smooth=True, outline="black", width=2)

        # Calculate the center of the rectangle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Add the text inside the rectangle
        self.canvas.create_text(center_x, center_y, text=name, fill="black", font=("Helvetica", text_size))

    def connectNodesBazier(self, from_x, from_y, to_x, to_y, through_x=-1, through_y=-1):
        x1 = int((from_x+1) * self.field_size_x - self.padding_x)
        y1 = int((from_y+0.5) * self.field_size_y)
        x2 = int(to_x * self.field_size_x + self.padding_x)
        y2 = int((to_y+0.5) * self.field_size_y)
        x3 = int((through_x+0.5) * self.field_size_x)
        y3 = int((through_y+0.5) * self.field_size_y)

        # Calculate control points for the Bezier curve
        control_x1 = x1 + (x2 - x1) / 3
        control_y1 = y1
        control_x2 = x1 + 2 * (x2 - x1) / 3
        control_y2 = y2

        # Create the curved line
        if (through_x == -1 or through_y == -1):
            self.canvas.create_line(x1, y1, control_x1, control_y1, control_x2, control_y2, x2, y2, smooth=True, width = 3)
        else:
            control_x1 = x1 + (x3 - x1) / 3
            control_y1 = y1
            control_x2 = x3 + 2 * (x2 - x3) / 3
            control_y2 = y2
            control_x3_in  = x1 + 2 * (x3 - x1) / 3
            control_y3_in  = y3
            control_x3_out = x3 + (x2 - x3) / 3
            control_y3_out = y3
            self.canvas.create_line(x1, y1, control_x1, control_y1, control_x3_in, control_y3_in, x3, y3, smooth=True, width = 3)
            self.canvas.create_line(x3, y3, control_x3_out, control_y3_out, control_x2, control_y2, x2, y2, smooth=True, width = 3)

        # Add arrowhead
        self.canvas.create_line(x2 - 10, y2 - 10, x2, y2, x2 - 10, y2 + 10, fill="black", width = 3)

    def connectNodesThreePart(self, from_x, from_y, to_x, to_y):
        x1 = int((from_x+1) * self.field_size_x - self.padding_x)
        y1 = int((from_y+0.5) * self.field_size_y)
        x2 = int(to_x * self.field_size_x + self.padding_x)
        y2 = int((to_y+0.5) * self.field_size_y)

        # Calculate the control points for the smooth bends
        vertical_x = int(to_x * self.field_size_x)
        vertical_y_dir = 1 if (from_y > to_y) else -1
        arc_rad = int(self.padding_x * 0.4)
        bend_x1 = vertical_x - arc_rad
        bend_y1 = y1
        bend_x2 = vertical_x
        bend_y2 = y1 - arc_rad * vertical_y_dir
        bend_x3 = vertical_x
        bend_y3 = y2 + arc_rad * vertical_y_dir
        bend_x4 = vertical_x + arc_rad
        bend_y4 = y2

        if (to_y == from_y):
            # Draw single line
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width = 3)
        else:
            # Draw lines and bends
            self.canvas.create_line(x1, y1, bend_x1, bend_y1, fill="black", width = 3)
            self.canvas.create_arc(bend_x1 - arc_rad, bend_y1, bend_x2, bend_y2 - arc_rad * vertical_y_dir, start=315-45*vertical_y_dir, extent=90, style=tk.ARC, width = 3)
            self.canvas.create_line(bend_x2, bend_y2, bend_x3, bend_y3, fill="black", width = 3)
            self.canvas.create_arc(bend_x3, bend_y3 + arc_rad * vertical_y_dir, bend_x4 + arc_rad, bend_y4, start=135-45*vertical_y_dir, extent=90, style=tk.ARC, width = 3)
            self.canvas.create_line(bend_x4, bend_y4, x2, y2, fill="black", width = 3)

        # Add arrowhead
        self.canvas.create_line(x2 - 10, y2 - 10, x2, y2, x2 - 10, y2 + 10, fill="black", width = 3)

    def mainloop(self):
        self.window_root.mainloop()

class graph:
    def __init__(self):
        pass

    def draw(self, grid, start_x, start_y):
        """
        Returns tuple (end_x, end_y, input_coords, output_coords)
        """
        return (start_x, start_y, [], [])

class graphSimple(graph):
    def __init__(self, name):
        self.name = name

    def draw(self, grid, start_x, start_y):
        grid.addNode(start_x, start_y, name = self.name, color = "#77DE60")
        return (start_x + 1, start_y + 1, [[start_x, start_y]], [[start_x, start_y]])

class graphSerial(graph):
    def __init__(self):
        self.sub_graphs = []

    def addStage(self, graph):
        assert (isinstance(graph, graphParallel) or isinstance(graph, graphSimple)), f"Serial graph may only contain simple or parallel sub-graphs"
        self.sub_graphs.append(graph)

    def draw(self, grid, start_x, start_y):
        x = start_x
        y = start_y
        end_y = start_y
        input_coords = []
        output_coords = []
        o_cs_pre = []
        for i,g in enumerate(self.sub_graphs):
            x, y, i_cs, o_cs = g.draw(grid, x, start_y)
            end_y = max(end_y, y)
            for o_c in o_cs_pre:
                for i_c in i_cs:
                    grid.connectNodesThreePart(o_c[0], o_c[1], i_c[0], i_c[1])
            o_cs_pre = o_cs
            if (i == 0):
                input_coords = i_cs
            if (i == len(self.sub_graphs)-1):
                output_coords = o_cs
        end_x = x
        return (end_x, end_y, input_coords, output_coords)

class graphParallel(graph):
    def __init__(self):
        self.sub_graphs = []

    def addStage(self, graph):
        assert (isinstance(graph, graphSerial) or isinstance(graph, graphSimple)), f"Parallel graph may only contain simple or serial sub-graphs"
        self.sub_graphs.append(graph)

    def draw(self, grid, start_x, start_y):
        x = start_x
        y = start_y
        end_x = start_x
        input_coords = []
        output_coords = []
        for i,g in enumerate(self.sub_graphs):
            x, y, i_cs, o_cs = g.draw(grid, start_x, y)
            end_x = max(end_x, x)
            input_coords += i_cs
            output_coords += o_cs
        end_y = y
        return (end_x, end_y, input_coords, output_coords)

def getGraphFromJson(json_data, is_parallel=False):
    graph = graphParallel() if (is_parallel) else graphSerial()
    for k, v in json_data.items():
        if (len(k) >= 6 and k[:6] == "serial"):
            graph.addStage(getGraphFromJson(v, False))
        elif (len(k) >= 8 and k[:8] == "parallel"):
            graph.addStage(getGraphFromJson(v, True))
        elif (len(k) >= 5 and k[:5] == "stage"):
            graph.addStage(graphSimple(v))
    return graph

if (__name__ == "__main__"):
    with open("input_test0.json", "r") as f:
        data = json.load(f)

    g = getGraphFromJson(data, True)
    grid = guiGraphGrid(14, 14, 200, 80, border_width=1, padding_x=40, padding_y=20)
    max_x, max_y, i_cs, o_cs = g.draw(grid, 0, 0)

    grid.mainloop()