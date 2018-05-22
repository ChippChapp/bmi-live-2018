"""Basic Model Interface (BMI) for the Diffusion model."""

import numpy as np
from basic_modeling_interface import Bmi
from .diffusion import Diffusion


class BmiDiffusion(Bmi):

    _name = 'Diffusion model'
    _input_var_names = ('plate_surface__temperature',)
    _output_var_names = ('plate_surface__temperature',)

    def __init__(self):
        self._model = None
        self._values = {}
        self._var_units = {}
        self._grids = {}
        self._grid_type = {}

    def initialize(self, filename=None):
        self._model = Diffusion(config_file=filename)

        self._values = {
            'plate_surface__temperature': self._model.temperature,
        }
        self._var_units = {
            'plate_surface__temperature': 'C'
        }
        self._grids = {
            0: ['plate_surface__temperature']
        }
        self._grid_type = {
            0: 'uniform_rectilinear_grid'
        }

    def update(self):
        return self._model.advance()

    def update_frac(self, time_frac):
		# fractional update
		# handy if models with different time steps are coupled
		time_step = self.get_time_step()
		self._model.dt = time_frac * time_step
		self.update()
		self._model.dt = time_step

    def update_until(self, then):
		n_steps = (then - self.get_current_time()) / self.get_time_step()
		
		for _ in range(int(n_steps)):
			self.update()
		
		if (n_steps - int(n_steps)) > 0.0:
			self.update_frac(n_steps - int(n_steps))

    def finalize(self):
        self._model = None

    def get_var_type(self, var_name):
        pass

    def get_var_units(self, var_name):
        pass

    def get_var_nbytes(self, var_name):
        pass

    def get_var_grid(self, var_name):
        pass

    def get_grid_rank(self, grid_id):
        pass

    def get_grid_size(self, grid_id):
        pass

    def get_value_ref(self, var_name):
		# BMI definition is to return flattened arrays
        return self._values[var_name].reshape(-1)

    def get_value(self, var_name):
        return self.get_value_ref(var_name).copy()

    def set_value(self, var_name, src):
		val = self.get_value_ref(var_name)
		val[:] = src
		
    def get_component_name(self):
        return self._name

    def get_input_var_names(self):
        return self._input_var_names

    def get_output_var_names(self):
        return self._output_var_names

    def get_grid_shape(self, grid_id):
        pass

    def get_grid_spacing(self, grid_id):
        pass

    def get_grid_origin(self, grid_id):
        pass

    def get_grid_type(self, grid_id):
        pass

    def get_start_time(self):
        return 0.0

    def get_end_time(self):
        return np.finfo('d').max

    def get_current_time(self):
        return self._model.time

    def get_time_step(self):
        return self._model.dt

    def get_time_units(self):
        return '-'
