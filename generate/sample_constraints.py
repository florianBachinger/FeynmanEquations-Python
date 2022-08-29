import json
import numpy as np
from sympy import parse_expr, diff, Symbol
from sympy import sqrt,exp,pi,asin,sin,acos,cos, tanh, ln, log
import Feynman.Functions as ff

# declaring a class
class obj:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

dict = []

i = 0
with open("Feynman/Constraints.py", "a") as text_file:
  text_file.write('constraints = [')
  #iterate over all equations
  for functionDictionary in ff.FunctionsJson:
    fObj = dict2obj(functionDictionary)

    dim = len(fObj.Variables)

    #generate uniform input space for range as specified in Feynman equations
    xs = np.random.uniform([v.low for v in fObj.Variables],
                            [v.high for v in fObj.Variables],
                            (10000, dim)
                        ).astype(float)
    print(f"{i} - Equation {fObj.EquationName} - {fObj.DescriptiveName}")
    i=i+1
    print(f"> {fObj.Formula_Str}")

    constraints = []
    local_variables = { v.name : Symbol(v.name) for v  in fObj.Variables }

    for var in [ v.name for v  in fObj.Variables]:

      formulaString  = fObj.Formula_Str
      formulaString = formulaString.replace('arcsin','asin')
      formulaString = formulaString.replace('arccos','acos')

  

      #let sympy parse expression and use local variables to ensure that no 
      # variable name is already in use as a keyword (e.g. gamma)
      equation=parse_expr(formulaString,local_variables)
  
      for order in range(1,3):
        #derive symbolically
        symb_deriv = diff(equation,var, order)
        print(f"> derive over {var} in order {order}: {symb_deriv}")

        #put derivative in lambda stump to be enable calculation
        diff_eq_lambda_str = fObj.Formula_Lambda_Stump.format(symb_deriv)
        diff_eq = eval(diff_eq_lambda_str)

        #calculate gradient per data point
        gradients = np.array([ diff_eq(row) for row in xs ])

        #sign of the first to determien descriptor
        sign_of_first = np.sign(gradients[0])

        if sign_of_first ==  -1:
            descriptor = "decreasing"
        if sign_of_first ==  0:
            descriptor = "constant"
        if sign_of_first ==  1:
            descriptor = "increasing"

        #do all gradients have the same sign
        if(np.all( np.sign(gradients))):
          print(f">> monotonic {descriptor}")

        constraints.append({'name':var,
              'order_derivative':order,
              'monotonicity':descriptor,
              'derivative': str(symb_deriv),
              'derivative_lambda': diff_eq_lambda_str})

    text_file.write(str( {'EquationName':fObj.EquationName,
          'DescriptiveName':fObj.DescriptiveName,
          'Constraints': constraints,
          'Variables': functionDictionary['Variables']
            }))
    text_file.write(',')

  text_file.write(']')

