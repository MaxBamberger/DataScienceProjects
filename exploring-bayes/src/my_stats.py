class PMF(object):

	def __init__(self, d={1: .5, 0: .5}):
		self.d = d


	def prob(self, val):
		return self.d[val]


	def set(self, key_val):

		key, val = key_val

		if key in self.d:
			self.d[key] = val

			factor = 1. / sum(self.d.values())

			# Iterate PMF and normalize
			for key, val in self.d.iteritems():
				self.d[key] = val * factor

		else:
			print 'Value does not already exist!'

	def print_pmf(self):
		print self.d.items()


if __name__ == '__main__':
	die = PMF({"1": 1./6, "2": 1./6, "3": 1./6,
	           "4": 1./6, "5": 1./6, "6": 1./6 })

    #=> 0.166
	die.prob("3")

	#=> [("1", 0.166), ("2", 0.166), ("3", 0.166), ("4", 0.166), ("5", 0.166), ("6", 0.166)]
	die.print_pmf()

	# weight the die, be sure to renormalize
	die.set(("2", 1./2))

	#=> [("1", 0.125), ("2", 0.375), ("3", 0.125), ("4", 0.125), ("5", 0.125), ("6", 0.125)]
	die.print_pmf()
