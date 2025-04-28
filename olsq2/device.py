import json
import pkgutil

def get_device_by_name(name, swap_duration):
    device_set_edge = { "qx" : [(0,2), (0,1), (1,2), (2,3), (2,4), (3,4)],
                        "ourense": [(0, 1), (1, 2), (1, 3), (3, 4)],
                       "sycamore": [(0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9), (4, 9), (4, 10), (5, 10), (5, 11),
                                    (6, 12), (6, 13), (7, 13), (7, 14), (8, 14), (8, 15), (9, 15), (9, 16), (10, 16), (10, 17), (11, 17),
                                    (12, 18), (13, 18), (13, 19), (14, 19), (14, 20), (15, 20), (15, 21), (16, 21), (16, 22), (17, 22), (17, 23),
                                    (18, 24), (18, 25), (19, 25), (19, 26), (20, 26), (20, 27), (21, 27), (21, 28), (22, 28), (22, 29), (23, 29),
                                    (24, 30), (25, 30), (25, 31), (26, 31), (26, 32), (27, 32), (27, 33), (28, 33), (28, 34), (29, 34), (29, 35),
                                    (30, 36), (30, 37), (31, 37), (31, 38), (32, 38), (32, 39), (33, 39), (33, 40), (34, 40), (34, 41), (35, 41),
                                    (36, 42), (37, 42), (37, 43), (38, 43), (38, 44), (39, 44), (39, 45), (40, 45), (40, 46), (41, 46), (41, 47),
                                    (42, 48), (42, 49), (43, 49), (43, 50), (44, 50), (44, 51), (45, 51), (45, 52), (46, 52), (46, 53), (47, 53)],
                       "rochester": [(0, 1), (1, 2), (2, 3), (3, 4),
                                     (0, 5), (4, 6), (5, 9), (6, 13),
                                     (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
                                     (7, 16), (11, 17), (15, 18), (16, 19), (17, 23), (18, 27),
                                     (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
                                     (21, 28), (25, 29), (28, 32), (29, 36),
                                     (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
                                     (30, 39), (34, 40), (38, 41), (39, 42), (40, 46), (41, 50),
                                     (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),
                                     (44, 51), (48, 52)],
                       "tokyo": [(0, 1), (1, 2), (2, 3), (3, 4),
                                 (0, 5), (1, 6), (1, 7), (2, 6), (2, 7), (3, 8), (3, 9), (4, 8), (4, 9),
                                 (5, 6), (6, 7), (7, 8), (8, 9),
                                 (5, 10), (5, 11), (6, 10), (6, 11), (7, 12), (7, 13), (8, 12), (8, 13), (9, 14),
                                 (10, 11), (11, 12), (12, 13), (13, 14),
                                 (10, 15), (11, 16), (11, 17), (12, 16), (12, 17), (13, 18), (13, 19), (14, 18), (14, 19),
                                 (15, 16), (16, 17), (17, 18), (18, 19)],
                       "aspen-4": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                                   (0, 8), (3, 11), (4, 12), (7, 15),
                                   (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)],
                        "eagle": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (12, 13),
                                   (0, 14), (14, 18), (4, 15), (15, 22), (8, 16), (16, 26), (12, 17), (17, 30),
                                   (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
                                   (20, 33), (33, 39), (24, 34), (34, 43), (28, 35), (35, 47), (32, 36), (36, 51),
                                   (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
                                   (37, 52), (52, 56), (41, 53), (53, 60), (45, 54), (54, 64), (49, 55), (55, 68),
                                   (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70),
                                   (58, 71), (71, 77), (62, 72), (72, 81), (66, 73), (73, 85), (70, 74), (74, 89),
                                   (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89),
                                   (75, 90), (90, 94), (79, 91), (91, 98), (83, 92), (92, 102), (87, 93), (93, 106),
                                   (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108),
                                   (96, 109), (100, 110), (110, 118), (104, 111), (111, 112), (108, 112), (112, 126),
                                   (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126)]
                       }
    
    device_set_qubit_num = {"qx": 5,
                        "ourense": 5,
                       "sycamore": 54,
                       "rochester": 53,
                       "tokyo": 20,
                       "aspen-4": 16,
                       "eagle": 127}
    
    device = qcdevice(name=name, nqubits=device_set_qubit_num[name],
                        connection=device_set_edge[name], swap_duration=swap_duration)
    return device


class qcdevice:
    """ QC device class.
    Contains the necessary parameters of the quantum hardware for OLSQ.
    """

    def __init__(self, name: str, nqubits: int = None, connection: list = None,
                 swap_duration: int = None, fmeas: list = None, 
                 fsingle: list = None, ftwo: list = None):
        """ Create a QC device.
        The user can either input the device parameters, or use existing
        ones stored in olsq/devices/ in json format (especially for
        duplicating paper results).  The parameters of existing devices 
        are overriden if inputs are provided.

        Args:
            name: name for the device.  If it starts with "default_",
                use existing device; otherwise, more parameters needed.
            nqubits: (optional) the number of physical qubits.
            connection: (optional) set of edges connecting qubits.
            swap_duration: (optional) how many time units a SWAP takes.
            fmeas: (optional) measurement fidelity of each qubit.
            fsingle: (optional) single-qubit gate fidelity of each qubit
            ftwo: (optional) two-qubit gate fidelity of each edge.

        Example:
            To use existing "defualt_ourense" device
            >>> dev = qcdevice(name="default_ourense")
            To set up a new device
            >>> dev = qcdevice(name="dev", nqubits=5,
                    connection=[(0, 1), (1, 2), (1, 3), (3, 4)],
                    swap_duration=3)
        """

        # typechecking for inputs
        if not isinstance(name, str):
            raise TypeError("name should be a string.")
        if nqubits is not None:
            if not isinstance(nqubits, int):
                raise TypeError("nqubits should be an integer.")
        if swap_duration is not None:
            if not isinstance(swap_duration, int):
                raise TypeError("swap_duration should be an integer.")
        
        if connection is not None:
            if not isinstance(connection, (list, tuple)):
                raise TypeError("connection should be a list or tuple.")
            else:
                for edge in connection:
                    if not isinstance(edge, (list, tuple)):
                        raise TypeError(f"{edge} is not a list or tuple.")
                    elif len(edge) != 2:
                        raise TypeError(f"{edge} does not connect two qubits.")
                    if not isinstance(edge[0], int):
                        raise TypeError(f"{edge[0]} is not an integer.")
                    if not isinstance(edge[1], int):
                        raise TypeError(f"{edge[1]} is not an integer.")
        
        if fmeas is not None:
            if not isinstance(fmeas, (list, tuple)):
                raise TypeError("fmeas should be a list or tuple.")
            else:
                for fmeas_i in fmeas:
                    if not isinstance(fmeas_i, (int, float)):
                        raise TypeError(f"{fmeas_i} is not a number.")
        if fsingle is not None:
            if not isinstance(fsingle, (list, tuple)):
                raise TypeError("fsingle should be a list or tuple.")
            else:
                for fsingle_i in fsingle:
                    if not isinstance(fsingle_i, (int, float)):
                        raise TypeError(f"{fsingle_i} is not a number.")
        if ftwo is not None:
            if not isinstance(ftwo, (list, tuple)):
                raise TypeError("ftwo should be a list or tuple.")
            else:
                for ftwo_i in ftwo:
                    if not isinstance(ftwo_i, (int, float)):
                        raise TypeError(f"{ftwo_i} is not a number.")
        
        if name.startswith("default_"):
            # use an existing device
            f = pkgutil.get_data(__name__, "devices/" + name + ".json")
            data = json.loads(f)
            self.name = data["name"]
            self.count_physical_qubit = data["count_physical_qubit"]
            self.list_qubit_edge = tuple( tuple(edge)
                                          for edge in data["list_qubit_edge"])
            self.swap_duration = data["swap_duration"]
            if "list_fidelity_measure" in data:
                self.list_fidelity_measure = \
                    tuple(data["list_fidelity_measure"])
            if "list_fidelity_single" in data:
                self.list_fidelity_single = tuple(data["list_fidelity_single"])
            if "list_fidelity_two" in data:
                self.list_fidelity_two = tuple(data["list_fidelity_two"])
        else:
            self.name = name
        
        # set parameters from inputs with value checking
        if nqubits is not None:
            self.count_physical_qubit = nqubits
        if "count_physical_qubit" not in self.__dict__:
            raise AttributeError("No physical qubit count specified.")

        if connection is not None:
            for edge in connection:
                if edge[0] < 0 or edge[0] >= self.count_physical_qubit:
                    raise ValueError( (f"{edge[0]} is outside of range "
                                       f"[0, {self.count_physical_qubit}).") )
                if edge[1] < 0 or edge[1] >= self.count_physical_qubit:
                    raise ValueError( (f"{edge[1]} is outside of range "
                                       f"[0, {self.count_physical_qubit}).") )
            self.list_qubit_edge = tuple( tuple(edge) for edge in connection)
        if "list_qubit_edge" not in self.__dict__:
            raise AttributeError("No edge set is specified.")
        
        if swap_duration is not None: 
            self.swap_duration = swap_duration
        else:
            self.swap_duration = 3
        
        if fmeas is not None:
            if len(fmeas) != self.count_physical_qubit:
                raise ValueError( ("fmeas should have "
                                   f"{self.count_physical_qubit} data.") )
            self.list_fidelity_measure = tuple(fmeas)
        if "list_fidelity_measure" not in self.__dict__:
            self.list_fidelity_measure = \
                tuple(1 for _ in range(self.count_physical_qubit))
        
        if fsingle is not None:
            if len(fsingle) != self.count_physical_qubit:
                raise ValueError( ("fsingle should have "
                                   f"{self.count_physical_qubit} data.") )
            self.list_fidelity_single = tuple(fsingle)
        if "list_fidelity_single" not in self.__dict__:
            self.list_fidelity_single = \
                tuple(1 for _ in range(self.count_physical_qubit))
        
        if ftwo is not None:
            if len(ftwo) != len(self.list_qubit_edge):
                raise ValueError( ("ftwo should have "
                                   f"{len(self.list_qubit_edge)} data.") )
            self.list_fidelity_two = tuple(ftwo)
        if "list_fidelity_two" not in self.__dict__:
            self.list_fidelity_two = \
                tuple(1 for _ in range(len(self.list_qubit_edge)))
