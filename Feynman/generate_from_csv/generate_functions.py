import pandas as pd

lines = []
dict = []

# read and prepare standard equations 
feynmanEquations = pd.read_csv('Feynman/src/FeynmanEquations.csv')
# filter to exclude the empty lines in FeynmanEquations.csv
feynmanEquations = feynmanEquations[feynmanEquations['Number']>=1]
feynmanEquations['FunctionName'] = [f'Feynman{int(number)}' for number in feynmanEquations['Number']]
feynmanEquations['DescriptiveName'] = [f'Feynman{int(number)}, Lecture {filename}' for number,filename in zip(feynmanEquations['Number'],feynmanEquations['Filename'])]

# read and prepare bonus equations 
bonusEquasions = pd.read_csv('Feynman/src/BonusEquations.csv')
# filter to exclude the empty lines in FeynmanEquations.csv
bonusEquasions = bonusEquasions[bonusEquasions['Number']>=1]
bonusEquasions['FunctionName'] = [f'Bonus{int(number)}' for number in bonusEquasions['Number']]
bonusEquasions['DescriptiveName'] = [f'Bonus{number}, {name}' for name,number in zip(bonusEquasions['Name'], bonusEquasions['Number'])]

#merge both into one dataframe
equations = feynmanEquations.append(bonusEquasions,ignore_index=True)

# add imports at file-beginning
lines.append('import pandas as pd')
numpyShort = 'np'
lines.append(f'import numpy as {numpyShort}')

lines.append("""
def Noise(target, noise_level = None):
  if( (noise_level == None) | (noise_level == 0)):
    return target
  assert 0 < noise_level < 1, f"Argument '{noise_level=}' out of range"

  stdDev = np.std(target)
  noise = np.random.normal(0,stdDev*np.sqrt(noise_level/(1-noise_level)),len(target))
  return target + noise
""")


for index, row in equations.iterrows():
  #iterate over the CSV rows
  size = 10000
  no_of_variables = int(row['# variables'])
  output = row['Output']
  formula = row['Formula']

  functionName = row['FunctionName']
  descriptiveName = row['DescriptiveName']
  
  print(f'{functionName}, {descriptiveName}')

  #format csv formulas to be numpy compatible
  formula_formatted = formula
  formula_formatted = formula_formatted.replace('arcsin',f'placeholder1')
  formula_formatted = formula_formatted.replace('arccos',f'placeholder2')

  formula_formatted = formula_formatted.replace('sqrt',f'{numpyShort}.sqrt')
  formula_formatted = formula_formatted.replace('exp',f'{numpyShort}.exp')
  formula_formatted = formula_formatted.replace('pi',f'{numpyShort}.pi')
  formula_formatted = formula_formatted.replace('sin',f'{numpyShort}.sin')
  formula_formatted = formula_formatted.replace('cos',f'{numpyShort}.cos')
  formula_formatted = formula_formatted.replace('tanh',f'{numpyShort}.tanh')
  formula_formatted = formula_formatted.replace('ln',f'{numpyShort}.ln')

  formula_formatted = formula_formatted.replace('placeholder1',f'{numpyShort}.arcsin')
  formula_formatted = formula_formatted.replace('placeholder2',f'{numpyShort}.arccos')

  #placeholders for 10 variables were added to the CSV
  #parsing each (even if the are empty) and adding them to a list
  variables = [
    ( row['v1_name'],row['v1_low'],row['v1_high'] ),
    ( row['v2_name'],row['v2_low'],row['v2_high'] ),
    ( row['v3_name'],row['v3_low'],row['v3_high'] ),
    ( row['v4_name'],row['v4_low'],row['v4_high'] ),
    ( row['v5_name'],row['v5_low'],row['v5_high'] ),
    ( row['v6_name'],row['v6_low'],row['v6_high'] ),
    ( row['v7_name'],row['v7_low'],row['v7_high'] ),
    ( row['v8_name'],row['v8_low'],row['v8_high'] ),
    ( row['v9_name'],row['v9_low'],row['v9_high'] ),
    ( row['v10_name'],row['v10_low'],row['v10_high'] ),
  ]
  #keeping only the number of variables used per row
  variables = variables[:no_of_variables]

  arguments = []
  names_string_commaSeparated = []
  variable_names_commaSeparated = []
  uniform_ranges = []
  variables_and_ranges = []

  for (name, low, high) in variables:
    #used for the function docstring
    arguments.append(f'        {name}: float or array-like, default range ({low},{high})')
    #used to generate the data for the Feynman functions
    uniform_ranges.append(f'    {name} = {numpyShort}.random.uniform({low},{high}, size)')
    #all variables and functions are also added to a dictionary to improve generic programmability
    variables_and_ranges.append({"name":name, "low":low , "high":high})
    #used for the dataframe definitions
    names_string_commaSeparated.append(f"'{name}'")
    #used for e.g. function parameters
    variable_names_commaSeparated.append(name)

  #add the resulting output variable name 
  names_string_commaSeparated.append(f"'{output}'")

  asDictionary = {'FunctionName':functionName,
                  'DescriptiveName':descriptiveName,
                  'Formula_Str':formula,
                  'Formula':formula_formatted,
                  'Variables':variables_and_ranges}
  dict.append( str(asDictionary))

  arguments = "\n".join(arguments)
  uniform_ranges = "\n".join(uniform_ranges)
  names_string_commaSeparated = ",".join(names_string_commaSeparated)
  variable_names_commaSeparated = ",".join(variable_names_commaSeparated)

  # multiline code-template used to generate the actual python code
  # first function only includes the formula itself and is resused by the 
  # data generation function returning a dataframe with inputs and target
  lines.append(f'''
def {functionName}_data(size = {size}, noise_level = 0):
    """
    {descriptiveName}

    Arguments:
        size: length of the inputs,
              sampled from the uniformly distributed standard ranges
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        pandas DataFrame [{names_string_commaSeparated}]
    """
{uniform_ranges}
    return {functionName}({variable_names_commaSeparated},noise_level)

def {functionName}({variable_names_commaSeparated}, noise_level = 0):
    """
    {descriptiveName}

    Arguments:
{arguments}
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: {formula}
    """
    target = {formula_formatted}
    return pd.DataFrame(
      list(
        zip(
          {variable_names_commaSeparated}
          ,Noise(target,noise_level)
        )
      )
      ,columns=[{names_string_commaSeparated}]
    )
  '''.format(functionName,
            size,
            descriptiveName,
            names_string_commaSeparated,
            uniform_ranges,
            variable_names_commaSeparated,
            arguments,
            formula,
            formula_formatted)
  )



#add descriptive dictionary
dict = ',\n'.join(dict)
dict = f'Functions = [\n{dict} \n ]'
lines.append(dict)

#write to file
with open("Feynman/Functions.py", "w") as text_file:
    text_file.write("\n".join(lines))