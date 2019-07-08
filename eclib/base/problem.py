
class BasicProblem(object):

	obj_types = []
	con_types = []

	def __init__(self, function):
		self.function = function

	def __call__(self, *args, **kwargs):
		return self.function(*args, **kwargs)


class Problem(BasicProblem):
	pass


class ConstraintProblem(Problem):
	pass
