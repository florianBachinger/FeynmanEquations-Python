import pandas as pd
import numpy as np

def Noise(target, noise_level = None):
  if( (noise_level == None) | (noise_level == 0)):
    return target
  assert 0 < noise_level < 1, f"Argument '{noise_level=}' out of range"

  stdDev = np.std(target)
  noise = np.random.normal(0,stdDev*np.sqrt(noise_level/(1-noise_level)),len(target))
  return target + noise


class Feynman1:
  equation_lambda = lambda args : (lambda theta: np.exp(-theta**2/2)/np.sqrt(2*np.pi) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman1, Lecture I.6.2a

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['theta','f']
      """
      theta = np.random.uniform(1.0,3.0, size)
      return Feynman1.calculate_df(theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(theta, noise_level = 0, include_original_target = False):
      """
      Feynman1, Lecture I.6.2a

      Arguments:
          theta: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['theta','f']
      """
      target = Feynman1.calculate(theta)
      data = [theta]
      data.append(Noise(target,noise_level))
      columns = ['theta','f']

      if(include_original_target):
         data.append(target)
         columns.append('f_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman1, Lecture I.6.2a

      Arguments:
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-theta**2/2)/sqrt(2*pi)
      """
      theta = X[0]
      return Feynman1.calculate(theta)

  @staticmethod
  def calculate(theta):
      """
      Feynman1, Lecture I.6.2a

      Arguments:
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-theta**2/2)/sqrt(2*pi)
      """
      return np.exp(-theta**2/2)/np.sqrt(2*np.pi)
  

class Feynman2:
  equation_lambda = lambda args : (lambda sigma,theta: np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman2, Lecture I.6.2

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma','theta','f']
      """
      sigma = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      return Feynman2.calculate_df(sigma,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(sigma,theta, noise_level = 0, include_original_target = False):
      """
      Feynman2, Lecture I.6.2

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma','theta','f']
      """
      target = Feynman2.calculate(sigma,theta)
      data = [sigma,theta]
      data.append(Noise(target,noise_level))
      columns = ['sigma','theta','f']

      if(include_original_target):
         data.append(target)
         columns.append('f_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman2, Lecture I.6.2

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)
      """
      sigma = X[0]
      theta = X[1]
      return Feynman2.calculate(sigma,theta)

  @staticmethod
  def calculate(sigma,theta):
      """
      Feynman2, Lecture I.6.2

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)
      """
      return np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
  

class Feynman3:
  equation_lambda = lambda args : (lambda sigma,theta,theta1: np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman3, Lecture I.6.2b

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma','theta','theta1','f']
      """
      sigma = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      theta1 = np.random.uniform(1.0,3.0, size)
      return Feynman3.calculate_df(sigma,theta,theta1,noise_level,include_original_target)

  @staticmethod
  def calculate_df(sigma,theta,theta1, noise_level = 0, include_original_target = False):
      """
      Feynman3, Lecture I.6.2b

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma','theta','theta1','f']
      """
      target = Feynman3.calculate(sigma,theta,theta1)
      data = [sigma,theta,theta1]
      data.append(Noise(target,noise_level))
      columns = ['sigma','theta','theta1','f']

      if(include_original_target):
         data.append(target)
         columns.append('f_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman3, Lecture I.6.2b

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)
      """
      sigma = X[0]
      theta = X[1]
      theta1 = X[2]
      return Feynman3.calculate(sigma,theta,theta1)

  @staticmethod
  def calculate(sigma,theta,theta1):
      """
      Feynman3, Lecture I.6.2b

      Arguments:
          sigma: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (1.0,3.0)
      Returns:
          f: exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)
      """
      return np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
  

class Feynman4:
  equation_lambda = lambda args : (lambda x1,x2,y1,y2: np.sqrt((x2-x1)**2+(y2-y1)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman4, Lecture I.8.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','y1','y2','d']
      """
      x1 = np.random.uniform(1.0,5.0, size)
      x2 = np.random.uniform(1.0,5.0, size)
      y1 = np.random.uniform(1.0,5.0, size)
      y2 = np.random.uniform(1.0,5.0, size)
      return Feynman4.calculate_df(x1,x2,y1,y2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x1,x2,y1,y2, noise_level = 0, include_original_target = False):
      """
      Feynman4, Lecture I.8.14

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','y1','y2','d']
      """
      target = Feynman4.calculate(x1,x2,y1,y2)
      data = [x1,x2,y1,y2]
      data.append(Noise(target,noise_level))
      columns = ['x1','x2','y1','y2','d']

      if(include_original_target):
         data.append(target)
         columns.append('d_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman4, Lecture I.8.14

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt((x2-x1)**2+(y2-y1)**2)
      """
      x1 = X[0]
      x2 = X[1]
      y1 = X[2]
      y2 = X[3]
      return Feynman4.calculate(x1,x2,y1,y2)

  @staticmethod
  def calculate(x1,x2,y1,y2):
      """
      Feynman4, Lecture I.8.14

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt((x2-x1)**2+(y2-y1)**2)
      """
      return np.sqrt((x2-x1)**2+(y2-y1)**2)
  

class Feynman5:
  equation_lambda = lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman5, Lecture I.9.18

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','G','x1','x2','y1','y2','z1','z2','F']
      """
      m1 = np.random.uniform(1.0,2.0, size)
      m2 = np.random.uniform(1.0,2.0, size)
      G = np.random.uniform(1.0,2.0, size)
      x1 = np.random.uniform(3.0,4.0, size)
      x2 = np.random.uniform(1.0,2.0, size)
      y1 = np.random.uniform(3.0,4.0, size)
      y2 = np.random.uniform(1.0,2.0, size)
      z1 = np.random.uniform(3.0,4.0, size)
      z2 = np.random.uniform(1.0,2.0, size)
      return Feynman5.calculate_df(m1,m2,G,x1,x2,y1,y2,z1,z2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m1,m2,G,x1,x2,y1,y2,z1,z2, noise_level = 0, include_original_target = False):
      """
      Feynman5, Lecture I.9.18

      Arguments:
          m1: float or array-like, default range (1.0,2.0)
          m2: float or array-like, default range (1.0,2.0)
          G: float or array-like, default range (1.0,2.0)
          x1: float or array-like, default range (3.0,4.0)
          x2: float or array-like, default range (1.0,2.0)
          y1: float or array-like, default range (3.0,4.0)
          y2: float or array-like, default range (1.0,2.0)
          z1: float or array-like, default range (3.0,4.0)
          z2: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','G','x1','x2','y1','y2','z1','z2','F']
      """
      target = Feynman5.calculate(m1,m2,G,x1,x2,y1,y2,z1,z2)
      data = [m1,m2,G,x1,x2,y1,y2,z1,z2]
      data.append(Noise(target,noise_level))
      columns = ['m1','m2','G','x1','x2','y1','y2','z1','z2','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman5, Lecture I.9.18

      Arguments:
          m1: float or array-like, default range (1.0,2.0)
          m2: float or array-like, default range (1.0,2.0)
          G: float or array-like, default range (1.0,2.0)
          x1: float or array-like, default range (3.0,4.0)
          x2: float or array-like, default range (1.0,2.0)
          y1: float or array-like, default range (3.0,4.0)
          y2: float or array-like, default range (1.0,2.0)
          z1: float or array-like, default range (3.0,4.0)
          z2: float or array-like, default range (1.0,2.0)
      Returns:
          f: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
      """
      m1 = X[0]
      m2 = X[1]
      G = X[2]
      x1 = X[3]
      x2 = X[4]
      y1 = X[5]
      y2 = X[6]
      z1 = X[7]
      z2 = X[8]
      return Feynman5.calculate(m1,m2,G,x1,x2,y1,y2,z1,z2)

  @staticmethod
  def calculate(m1,m2,G,x1,x2,y1,y2,z1,z2):
      """
      Feynman5, Lecture I.9.18

      Arguments:
          m1: float or array-like, default range (1.0,2.0)
          m2: float or array-like, default range (1.0,2.0)
          G: float or array-like, default range (1.0,2.0)
          x1: float or array-like, default range (3.0,4.0)
          x2: float or array-like, default range (1.0,2.0)
          y1: float or array-like, default range (3.0,4.0)
          y2: float or array-like, default range (1.0,2.0)
          z1: float or array-like, default range (3.0,4.0)
          z2: float or array-like, default range (1.0,2.0)
      Returns:
          f: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
      """
      return G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
  

class Feynman6:
  equation_lambda = lambda args : (lambda m_0,v,c: m_0/np.sqrt(1-v**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman6, Lecture I.10.7

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m_0','v','c','m']
      """
      m_0 = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman6.calculate_df(m_0,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m_0,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman6, Lecture I.10.7

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m_0','v','c','m']
      """
      target = Feynman6.calculate(m_0,v,c)
      data = [m_0,v,c]
      data.append(Noise(target,noise_level))
      columns = ['m_0','v','c','m']

      if(include_original_target):
         data.append(target)
         columns.append('m_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman6, Lecture I.10.7

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m_0/sqrt(1-v**2/c**2)
      """
      m_0 = X[0]
      v = X[1]
      c = X[2]
      return Feynman6.calculate(m_0,v,c)

  @staticmethod
  def calculate(m_0,v,c):
      """
      Feynman6, Lecture I.10.7

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m_0/sqrt(1-v**2/c**2)
      """
      return m_0/np.sqrt(1-v**2/c**2)
  

class Feynman7:
  equation_lambda = lambda args : (lambda x1,x2,x3,y1,y2,y3: x1*y1+x2*y2+x3*y3 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman7, Lecture I.11.19

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','x3','y1','y2','y3','A']
      """
      x1 = np.random.uniform(1.0,5.0, size)
      x2 = np.random.uniform(1.0,5.0, size)
      x3 = np.random.uniform(1.0,5.0, size)
      y1 = np.random.uniform(1.0,5.0, size)
      y2 = np.random.uniform(1.0,5.0, size)
      y3 = np.random.uniform(1.0,5.0, size)
      return Feynman7.calculate_df(x1,x2,x3,y1,y2,y3,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x1,x2,x3,y1,y2,y3, noise_level = 0, include_original_target = False):
      """
      Feynman7, Lecture I.11.19

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          x3: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
          y3: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','x3','y1','y2','y3','A']
      """
      target = Feynman7.calculate(x1,x2,x3,y1,y2,y3)
      data = [x1,x2,x3,y1,y2,y3]
      data.append(Noise(target,noise_level))
      columns = ['x1','x2','x3','y1','y2','y3','A']

      if(include_original_target):
         data.append(target)
         columns.append('A_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman7, Lecture I.11.19

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          x3: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
          y3: float or array-like, default range (1.0,5.0)
      Returns:
          f: x1*y1+x2*y2+x3*y3
      """
      x1 = X[0]
      x2 = X[1]
      x3 = X[2]
      y1 = X[3]
      y2 = X[4]
      y3 = X[5]
      return Feynman7.calculate(x1,x2,x3,y1,y2,y3)

  @staticmethod
  def calculate(x1,x2,x3,y1,y2,y3):
      """
      Feynman7, Lecture I.11.19

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          x3: float or array-like, default range (1.0,5.0)
          y1: float or array-like, default range (1.0,5.0)
          y2: float or array-like, default range (1.0,5.0)
          y3: float or array-like, default range (1.0,5.0)
      Returns:
          f: x1*y1+x2*y2+x3*y3
      """
      return x1*y1+x2*y2+x3*y3
  

class Feynman8:
  equation_lambda = lambda args : (lambda mu,Nn: mu*Nn )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman8, Lecture I.12.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mu','Nn','F']
      """
      mu = np.random.uniform(1.0,5.0, size)
      Nn = np.random.uniform(1.0,5.0, size)
      return Feynman8.calculate_df(mu,Nn,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mu,Nn, noise_level = 0, include_original_target = False):
      """
      Feynman8, Lecture I.12.1

      Arguments:
          mu: float or array-like, default range (1.0,5.0)
          Nn: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mu','Nn','F']
      """
      target = Feynman8.calculate(mu,Nn)
      data = [mu,Nn]
      data.append(Noise(target,noise_level))
      columns = ['mu','Nn','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman8, Lecture I.12.1

      Arguments:
          mu: float or array-like, default range (1.0,5.0)
          Nn: float or array-like, default range (1.0,5.0)
      Returns:
          f: mu*Nn
      """
      mu = X[0]
      Nn = X[1]
      return Feynman8.calculate(mu,Nn)

  @staticmethod
  def calculate(mu,Nn):
      """
      Feynman8, Lecture I.12.1

      Arguments:
          mu: float or array-like, default range (1.0,5.0)
          Nn: float or array-like, default range (1.0,5.0)
      Returns:
          f: mu*Nn
      """
      return mu*Nn
  

class Feynman10:
  equation_lambda = lambda args : (lambda q1,q2,epsilon,r: q1*q2*r/(4*np.pi*epsilon*r**3) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman10, Lecture I.12.2

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q1','q2','epsilon','r','F']
      """
      q1 = np.random.uniform(1.0,5.0, size)
      q2 = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman10.calculate_df(q1,q2,epsilon,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q1,q2,epsilon,r, noise_level = 0, include_original_target = False):
      """
      Feynman10, Lecture I.12.2

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          q2: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q1','q2','epsilon','r','F']
      """
      target = Feynman10.calculate(q1,q2,epsilon,r)
      data = [q1,q2,epsilon,r]
      data.append(Noise(target,noise_level))
      columns = ['q1','q2','epsilon','r','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman10, Lecture I.12.2

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          q2: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q1*q2*r/(4*pi*epsilon*r**3)
      """
      q1 = X[0]
      q2 = X[1]
      epsilon = X[2]
      r = X[3]
      return Feynman10.calculate(q1,q2,epsilon,r)

  @staticmethod
  def calculate(q1,q2,epsilon,r):
      """
      Feynman10, Lecture I.12.2

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          q2: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q1*q2*r/(4*pi*epsilon*r**3)
      """
      return q1*q2*r/(4*np.pi*epsilon*r**3)
  

class Feynman11:
  equation_lambda = lambda args : (lambda q1,epsilon,r: q1*r/(4*np.pi*epsilon*r**3) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman11, Lecture I.12.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q1','epsilon','r','Ef']
      """
      q1 = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman11.calculate_df(q1,epsilon,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q1,epsilon,r, noise_level = 0, include_original_target = False):
      """
      Feynman11, Lecture I.12.4

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q1','epsilon','r','Ef']
      """
      target = Feynman11.calculate(q1,epsilon,r)
      data = [q1,epsilon,r]
      data.append(Noise(target,noise_level))
      columns = ['q1','epsilon','r','Ef']

      if(include_original_target):
         data.append(target)
         columns.append('Ef_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman11, Lecture I.12.4

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q1*r/(4*pi*epsilon*r**3)
      """
      q1 = X[0]
      epsilon = X[1]
      r = X[2]
      return Feynman11.calculate(q1,epsilon,r)

  @staticmethod
  def calculate(q1,epsilon,r):
      """
      Feynman11, Lecture I.12.4

      Arguments:
          q1: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q1*r/(4*pi*epsilon*r**3)
      """
      return q1*r/(4*np.pi*epsilon*r**3)
  

class Feynman12:
  equation_lambda = lambda args : (lambda q2,Ef: q2*Ef )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman12, Lecture I.12.5

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q2','Ef','F']
      """
      q2 = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      return Feynman12.calculate_df(q2,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q2,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman12, Lecture I.12.5

      Arguments:
          q2: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q2','Ef','F']
      """
      target = Feynman12.calculate(q2,Ef)
      data = [q2,Ef]
      data.append(Noise(target,noise_level))
      columns = ['q2','Ef','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman12, Lecture I.12.5

      Arguments:
          q2: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: q2*Ef
      """
      q2 = X[0]
      Ef = X[1]
      return Feynman12.calculate(q2,Ef)

  @staticmethod
  def calculate(q2,Ef):
      """
      Feynman12, Lecture I.12.5

      Arguments:
          q2: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: q2*Ef
      """
      return q2*Ef
  

class Feynman13:
  equation_lambda = lambda args : (lambda q,Ef,B,v,theta: q*(Ef+B*v*np.sin(theta)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman13, Lecture I.12.11

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','Ef','B','v','theta','F']
      """
      q = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      return Feynman13.calculate_df(q,Ef,B,v,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,Ef,B,v,theta, noise_level = 0, include_original_target = False):
      """
      Feynman13, Lecture I.12.11

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','Ef','B','v','theta','F']
      """
      target = Feynman13.calculate(q,Ef,B,v,theta)
      data = [q,Ef,B,v,theta]
      data.append(Noise(target,noise_level))
      columns = ['q','Ef','B','v','theta','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman13, Lecture I.12.11

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*(Ef+B*v*sin(theta))
      """
      q = X[0]
      Ef = X[1]
      B = X[2]
      v = X[3]
      theta = X[4]
      return Feynman13.calculate(q,Ef,B,v,theta)

  @staticmethod
  def calculate(q,Ef,B,v,theta):
      """
      Feynman13, Lecture I.12.11

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*(Ef+B*v*sin(theta))
      """
      return q*(Ef+B*v*np.sin(theta))
  

class Feynman9:
  equation_lambda = lambda args : (lambda m,v,u,w: 1/2*m*(v**2+u**2+w**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman9, Lecture I.13.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','v','u','w','K']
      """
      m = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      u = np.random.uniform(1.0,5.0, size)
      w = np.random.uniform(1.0,5.0, size)
      return Feynman9.calculate_df(m,v,u,w,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,v,u,w, noise_level = 0, include_original_target = False):
      """
      Feynman9, Lecture I.13.4

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
          w: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','v','u','w','K']
      """
      target = Feynman9.calculate(m,v,u,w)
      data = [m,v,u,w]
      data.append(Noise(target,noise_level))
      columns = ['m','v','u','w','K']

      if(include_original_target):
         data.append(target)
         columns.append('K_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman9, Lecture I.13.4

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
          w: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/2*m*(v**2+u**2+w**2)
      """
      m = X[0]
      v = X[1]
      u = X[2]
      w = X[3]
      return Feynman9.calculate(m,v,u,w)

  @staticmethod
  def calculate(m,v,u,w):
      """
      Feynman9, Lecture I.13.4

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
          w: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/2*m*(v**2+u**2+w**2)
      """
      return 1/2*m*(v**2+u**2+w**2)
  

class Feynman14:
  equation_lambda = lambda args : (lambda m1,m2,r1,r2,G: G*m1*m2*(1/r2-1/r1) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman14, Lecture I.13.12

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','r1','r2','G','U']
      """
      m1 = np.random.uniform(1.0,5.0, size)
      m2 = np.random.uniform(1.0,5.0, size)
      r1 = np.random.uniform(1.0,5.0, size)
      r2 = np.random.uniform(1.0,5.0, size)
      G = np.random.uniform(1.0,5.0, size)
      return Feynman14.calculate_df(m1,m2,r1,r2,G,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m1,m2,r1,r2,G, noise_level = 0, include_original_target = False):
      """
      Feynman14, Lecture I.13.12

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
          G: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','r1','r2','G','U']
      """
      target = Feynman14.calculate(m1,m2,r1,r2,G)
      data = [m1,m2,r1,r2,G]
      data.append(Noise(target,noise_level))
      columns = ['m1','m2','r1','r2','G','U']

      if(include_original_target):
         data.append(target)
         columns.append('U_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman14, Lecture I.13.12

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
          G: float or array-like, default range (1.0,5.0)
      Returns:
          f: G*m1*m2*(1/r2-1/r1)
      """
      m1 = X[0]
      m2 = X[1]
      r1 = X[2]
      r2 = X[3]
      G = X[4]
      return Feynman14.calculate(m1,m2,r1,r2,G)

  @staticmethod
  def calculate(m1,m2,r1,r2,G):
      """
      Feynman14, Lecture I.13.12

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
          G: float or array-like, default range (1.0,5.0)
      Returns:
          f: G*m1*m2*(1/r2-1/r1)
      """
      return G*m1*m2*(1/r2-1/r1)
  

class Feynman15:
  equation_lambda = lambda args : (lambda m,g,z: m*g*z )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman15, Lecture I.14.3

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','g','z','U']
      """
      m = np.random.uniform(1.0,5.0, size)
      g = np.random.uniform(1.0,5.0, size)
      z = np.random.uniform(1.0,5.0, size)
      return Feynman15.calculate_df(m,g,z,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,g,z, noise_level = 0, include_original_target = False):
      """
      Feynman15, Lecture I.14.3

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          z: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','g','z','U']
      """
      target = Feynman15.calculate(m,g,z)
      data = [m,g,z]
      data.append(Noise(target,noise_level))
      columns = ['m','g','z','U']

      if(include_original_target):
         data.append(target)
         columns.append('U_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman15, Lecture I.14.3

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          z: float or array-like, default range (1.0,5.0)
      Returns:
          f: m*g*z
      """
      m = X[0]
      g = X[1]
      z = X[2]
      return Feynman15.calculate(m,g,z)

  @staticmethod
  def calculate(m,g,z):
      """
      Feynman15, Lecture I.14.3

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          z: float or array-like, default range (1.0,5.0)
      Returns:
          f: m*g*z
      """
      return m*g*z
  

class Feynman16:
  equation_lambda = lambda args : (lambda k_spring,x: 1/2*k_spring*x**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman16, Lecture I.14.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['k_spring','x','U']
      """
      k_spring = np.random.uniform(1.0,5.0, size)
      x = np.random.uniform(1.0,5.0, size)
      return Feynman16.calculate_df(k_spring,x,noise_level,include_original_target)

  @staticmethod
  def calculate_df(k_spring,x, noise_level = 0, include_original_target = False):
      """
      Feynman16, Lecture I.14.4

      Arguments:
          k_spring: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['k_spring','x','U']
      """
      target = Feynman16.calculate(k_spring,x)
      data = [k_spring,x]
      data.append(Noise(target,noise_level))
      columns = ['k_spring','x','U']

      if(include_original_target):
         data.append(target)
         columns.append('U_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman16, Lecture I.14.4

      Arguments:
          k_spring: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/2*k_spring*x**2
      """
      k_spring = X[0]
      x = X[1]
      return Feynman16.calculate(k_spring,x)

  @staticmethod
  def calculate(k_spring,x):
      """
      Feynman16, Lecture I.14.4

      Arguments:
          k_spring: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/2*k_spring*x**2
      """
      return 1/2*k_spring*x**2
  

class Feynman17:
  equation_lambda = lambda args : (lambda x,u,c,t: (x-u*t)/np.sqrt(1-u**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman17, Lecture I.15.3x

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x','u','c','t','x1']
      """
      x = np.random.uniform(5.0,10.0, size)
      u = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,20.0, size)
      t = np.random.uniform(1.0,2.0, size)
      return Feynman17.calculate_df(x,u,c,t,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x,u,c,t, noise_level = 0, include_original_target = False):
      """
      Feynman17, Lecture I.15.3x

      Arguments:
          x: float or array-like, default range (5.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,20.0)
          t: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x','u','c','t','x1']
      """
      target = Feynman17.calculate(x,u,c,t)
      data = [x,u,c,t]
      data.append(Noise(target,noise_level))
      columns = ['x','u','c','t','x1']

      if(include_original_target):
         data.append(target)
         columns.append('x1_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman17, Lecture I.15.3x

      Arguments:
          x: float or array-like, default range (5.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,20.0)
          t: float or array-like, default range (1.0,2.0)
      Returns:
          f: (x-u*t)/sqrt(1-u**2/c**2)
      """
      x = X[0]
      u = X[1]
      c = X[2]
      t = X[3]
      return Feynman17.calculate(x,u,c,t)

  @staticmethod
  def calculate(x,u,c,t):
      """
      Feynman17, Lecture I.15.3x

      Arguments:
          x: float or array-like, default range (5.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,20.0)
          t: float or array-like, default range (1.0,2.0)
      Returns:
          f: (x-u*t)/sqrt(1-u**2/c**2)
      """
      return (x-u*t)/np.sqrt(1-u**2/c**2)
  

class Feynman18:
  equation_lambda = lambda args : (lambda x,c,u,t: (t-u*x/c**2)/np.sqrt(1-u**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman18, Lecture I.15.3t

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x','c','u','t','t1']
      """
      x = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(3.0,10.0, size)
      u = np.random.uniform(1.0,2.0, size)
      t = np.random.uniform(1.0,5.0, size)
      return Feynman18.calculate_df(x,c,u,t,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x,c,u,t, noise_level = 0, include_original_target = False):
      """
      Feynman18, Lecture I.15.3t

      Arguments:
          x: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (3.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x','c','u','t','t1']
      """
      target = Feynman18.calculate(x,c,u,t)
      data = [x,c,u,t]
      data.append(Noise(target,noise_level))
      columns = ['x','c','u','t','t1']

      if(include_original_target):
         data.append(target)
         columns.append('t1_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman18, Lecture I.15.3t

      Arguments:
          x: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (3.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,5.0)
      Returns:
          f: (t-u*x/c**2)/sqrt(1-u**2/c**2)
      """
      x = X[0]
      c = X[1]
      u = X[2]
      t = X[3]
      return Feynman18.calculate(x,c,u,t)

  @staticmethod
  def calculate(x,c,u,t):
      """
      Feynman18, Lecture I.15.3t

      Arguments:
          x: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (3.0,10.0)
          u: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,5.0)
      Returns:
          f: (t-u*x/c**2)/sqrt(1-u**2/c**2)
      """
      return (t-u*x/c**2)/np.sqrt(1-u**2/c**2)
  

class Feynman19:
  equation_lambda = lambda args : (lambda m_0,v,c: m_0*v/np.sqrt(1-v**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman19, Lecture I.15.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m_0','v','c','p']
      """
      m_0 = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman19.calculate_df(m_0,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m_0,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman19, Lecture I.15.1

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m_0','v','c','p']
      """
      target = Feynman19.calculate(m_0,v,c)
      data = [m_0,v,c]
      data.append(Noise(target,noise_level))
      columns = ['m_0','v','c','p']

      if(include_original_target):
         data.append(target)
         columns.append('p_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman19, Lecture I.15.1

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m_0*v/sqrt(1-v**2/c**2)
      """
      m_0 = X[0]
      v = X[1]
      c = X[2]
      return Feynman19.calculate(m_0,v,c)

  @staticmethod
  def calculate(m_0,v,c):
      """
      Feynman19, Lecture I.15.1

      Arguments:
          m_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m_0*v/sqrt(1-v**2/c**2)
      """
      return m_0*v/np.sqrt(1-v**2/c**2)
  

class Feynman20:
  equation_lambda = lambda args : (lambda c,v,u: (u+v)/(1+u*v/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman20, Lecture I.16.6

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','u','v1']
      """
      c = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      u = np.random.uniform(1.0,5.0, size)
      return Feynman20.calculate_df(c,v,u,noise_level,include_original_target)

  @staticmethod
  def calculate_df(c,v,u, noise_level = 0, include_original_target = False):
      """
      Feynman20, Lecture I.16.6

      Arguments:
          c: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','u','v1']
      """
      target = Feynman20.calculate(c,v,u)
      data = [c,v,u]
      data.append(Noise(target,noise_level))
      columns = ['c','v','u','v1']

      if(include_original_target):
         data.append(target)
         columns.append('v1_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman20, Lecture I.16.6

      Arguments:
          c: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
      Returns:
          f: (u+v)/(1+u*v/c**2)
      """
      c = X[0]
      v = X[1]
      u = X[2]
      return Feynman20.calculate(c,v,u)

  @staticmethod
  def calculate(c,v,u):
      """
      Feynman20, Lecture I.16.6

      Arguments:
          c: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          u: float or array-like, default range (1.0,5.0)
      Returns:
          f: (u+v)/(1+u*v/c**2)
      """
      return (u+v)/(1+u*v/c**2)
  

class Feynman21:
  equation_lambda = lambda args : (lambda m1,m2,r1,r2: (m1*r1+m2*r2)/(m1+m2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman21, Lecture I.18.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','r1','r2','r']
      """
      m1 = np.random.uniform(1.0,5.0, size)
      m2 = np.random.uniform(1.0,5.0, size)
      r1 = np.random.uniform(1.0,5.0, size)
      r2 = np.random.uniform(1.0,5.0, size)
      return Feynman21.calculate_df(m1,m2,r1,r2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m1,m2,r1,r2, noise_level = 0, include_original_target = False):
      """
      Feynman21, Lecture I.18.4

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m1','m2','r1','r2','r']
      """
      target = Feynman21.calculate(m1,m2,r1,r2)
      data = [m1,m2,r1,r2]
      data.append(Noise(target,noise_level))
      columns = ['m1','m2','r1','r2','r']

      if(include_original_target):
         data.append(target)
         columns.append('r_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman21, Lecture I.18.4

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
      Returns:
          f: (m1*r1+m2*r2)/(m1+m2)
      """
      m1 = X[0]
      m2 = X[1]
      r1 = X[2]
      r2 = X[3]
      return Feynman21.calculate(m1,m2,r1,r2)

  @staticmethod
  def calculate(m1,m2,r1,r2):
      """
      Feynman21, Lecture I.18.4

      Arguments:
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r1: float or array-like, default range (1.0,5.0)
          r2: float or array-like, default range (1.0,5.0)
      Returns:
          f: (m1*r1+m2*r2)/(m1+m2)
      """
      return (m1*r1+m2*r2)/(m1+m2)
  

class Feynman22:
  equation_lambda = lambda args : (lambda r,F,theta: r*F*np.sin(theta) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman22, Lecture I.18.12

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['r','F','theta','tau']
      """
      r = np.random.uniform(1.0,5.0, size)
      F = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(0.0,5.0, size)
      return Feynman22.calculate_df(r,F,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(r,F,theta, noise_level = 0, include_original_target = False):
      """
      Feynman22, Lecture I.18.12

      Arguments:
          r: float or array-like, default range (1.0,5.0)
          F: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['r','F','theta','tau']
      """
      target = Feynman22.calculate(r,F,theta)
      data = [r,F,theta]
      data.append(Noise(target,noise_level))
      columns = ['r','F','theta','tau']

      if(include_original_target):
         data.append(target)
         columns.append('tau_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman22, Lecture I.18.12

      Arguments:
          r: float or array-like, default range (1.0,5.0)
          F: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,5.0)
      Returns:
          f: r*F*sin(theta)
      """
      r = X[0]
      F = X[1]
      theta = X[2]
      return Feynman22.calculate(r,F,theta)

  @staticmethod
  def calculate(r,F,theta):
      """
      Feynman22, Lecture I.18.12

      Arguments:
          r: float or array-like, default range (1.0,5.0)
          F: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,5.0)
      Returns:
          f: r*F*sin(theta)
      """
      return r*F*np.sin(theta)
  

class Feynman23:
  equation_lambda = lambda args : (lambda m,r,v,theta: m*r*v*np.sin(theta) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman23, Lecture I.18.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','r','v','theta','L']
      """
      m = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      return Feynman23.calculate_df(m,r,v,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,r,v,theta, noise_level = 0, include_original_target = False):
      """
      Feynman23, Lecture I.18.14

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','r','v','theta','L']
      """
      target = Feynman23.calculate(m,r,v,theta)
      data = [m,r,v,theta]
      data.append(Noise(target,noise_level))
      columns = ['m','r','v','theta','L']

      if(include_original_target):
         data.append(target)
         columns.append('L_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman23, Lecture I.18.14

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: m*r*v*sin(theta)
      """
      m = X[0]
      r = X[1]
      v = X[2]
      theta = X[3]
      return Feynman23.calculate(m,r,v,theta)

  @staticmethod
  def calculate(m,r,v,theta):
      """
      Feynman23, Lecture I.18.14

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: m*r*v*sin(theta)
      """
      return m*r*v*np.sin(theta)
  

class Feynman24:
  equation_lambda = lambda args : (lambda m,omega,omega_0,x: 1/2*m*(omega**2+omega_0**2)*1/2*x**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman24, Lecture I.24.6

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','omega','omega_0','x','E_n']
      """
      m = np.random.uniform(1.0,3.0, size)
      omega = np.random.uniform(1.0,3.0, size)
      omega_0 = np.random.uniform(1.0,3.0, size)
      x = np.random.uniform(1.0,3.0, size)
      return Feynman24.calculate_df(m,omega,omega_0,x,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,omega,omega_0,x, noise_level = 0, include_original_target = False):
      """
      Feynman24, Lecture I.24.6

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','omega','omega_0','x','E_n']
      """
      target = Feynman24.calculate(m,omega,omega_0,x)
      data = [m,omega,omega_0,x]
      data.append(Noise(target,noise_level))
      columns = ['m','omega','omega_0','x','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman24, Lecture I.24.6

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
      Returns:
          f: 1/2*m*(omega**2+omega_0**2)*1/2*x**2
      """
      m = X[0]
      omega = X[1]
      omega_0 = X[2]
      x = X[3]
      return Feynman24.calculate(m,omega,omega_0,x)

  @staticmethod
  def calculate(m,omega,omega_0,x):
      """
      Feynman24, Lecture I.24.6

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
      Returns:
          f: 1/2*m*(omega**2+omega_0**2)*1/2*x**2
      """
      return 1/2*m*(omega**2+omega_0**2)*1/2*x**2
  

class Feynman25:
  equation_lambda = lambda args : (lambda q,C: q/C )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman25, Lecture I.25.13

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','C','Volt']
      """
      q = np.random.uniform(1.0,5.0, size)
      C = np.random.uniform(1.0,5.0, size)
      return Feynman25.calculate_df(q,C,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,C, noise_level = 0, include_original_target = False):
      """
      Feynman25, Lecture I.25.13

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          C: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','C','Volt']
      """
      target = Feynman25.calculate(q,C)
      data = [q,C]
      data.append(Noise(target,noise_level))
      columns = ['q','C','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman25, Lecture I.25.13

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          C: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/C
      """
      q = X[0]
      C = X[1]
      return Feynman25.calculate(q,C)

  @staticmethod
  def calculate(q,C):
      """
      Feynman25, Lecture I.25.13

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          C: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/C
      """
      return q/C
  

class Feynman26:
  equation_lambda = lambda args : (lambda n,theta2: np.arcsin(n*np.sin(theta2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman26, Lecture I.26.2

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','theta2','theta1']
      """
      n = np.random.uniform(0.0,1.0, size)
      theta2 = np.random.uniform(1.0,5.0, size)
      return Feynman26.calculate_df(n,theta2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,theta2, noise_level = 0, include_original_target = False):
      """
      Feynman26, Lecture I.26.2

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          theta2: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','theta2','theta1']
      """
      target = Feynman26.calculate(n,theta2)
      data = [n,theta2]
      data.append(Noise(target,noise_level))
      columns = ['n','theta2','theta1']

      if(include_original_target):
         data.append(target)
         columns.append('theta1_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman26, Lecture I.26.2

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          theta2: float or array-like, default range (1.0,5.0)
      Returns:
          f: arcsin(n*sin(theta2))
      """
      n = X[0]
      theta2 = X[1]
      return Feynman26.calculate(n,theta2)

  @staticmethod
  def calculate(n,theta2):
      """
      Feynman26, Lecture I.26.2

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          theta2: float or array-like, default range (1.0,5.0)
      Returns:
          f: arcsin(n*sin(theta2))
      """
      return np.arcsin(n*np.sin(theta2))
  

class Feynman27:
  equation_lambda = lambda args : (lambda d1,d2,n: 1/(1/d1+n/d2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman27, Lecture I.27.6

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d1','d2','n','foc']
      """
      d1 = np.random.uniform(1.0,5.0, size)
      d2 = np.random.uniform(1.0,5.0, size)
      n = np.random.uniform(1.0,5.0, size)
      return Feynman27.calculate_df(d1,d2,n,noise_level,include_original_target)

  @staticmethod
  def calculate_df(d1,d2,n, noise_level = 0, include_original_target = False):
      """
      Feynman27, Lecture I.27.6

      Arguments:
          d1: float or array-like, default range (1.0,5.0)
          d2: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d1','d2','n','foc']
      """
      target = Feynman27.calculate(d1,d2,n)
      data = [d1,d2,n]
      data.append(Noise(target,noise_level))
      columns = ['d1','d2','n','foc']

      if(include_original_target):
         data.append(target)
         columns.append('foc_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman27, Lecture I.27.6

      Arguments:
          d1: float or array-like, default range (1.0,5.0)
          d2: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(1/d1+n/d2)
      """
      d1 = X[0]
      d2 = X[1]
      n = X[2]
      return Feynman27.calculate(d1,d2,n)

  @staticmethod
  def calculate(d1,d2,n):
      """
      Feynman27, Lecture I.27.6

      Arguments:
          d1: float or array-like, default range (1.0,5.0)
          d2: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(1/d1+n/d2)
      """
      return 1/(1/d1+n/d2)
  

class Feynman28:
  equation_lambda = lambda args : (lambda omega,c: omega/c )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman28, Lecture I.29.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','c','k']
      """
      omega = np.random.uniform(1.0,10.0, size)
      c = np.random.uniform(1.0,10.0, size)
      return Feynman28.calculate_df(omega,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(omega,c, noise_level = 0, include_original_target = False):
      """
      Feynman28, Lecture I.29.4

      Arguments:
          omega: float or array-like, default range (1.0,10.0)
          c: float or array-like, default range (1.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','c','k']
      """
      target = Feynman28.calculate(omega,c)
      data = [omega,c]
      data.append(Noise(target,noise_level))
      columns = ['omega','c','k']

      if(include_original_target):
         data.append(target)
         columns.append('k_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman28, Lecture I.29.4

      Arguments:
          omega: float or array-like, default range (1.0,10.0)
          c: float or array-like, default range (1.0,10.0)
      Returns:
          f: omega/c
      """
      omega = X[0]
      c = X[1]
      return Feynman28.calculate(omega,c)

  @staticmethod
  def calculate(omega,c):
      """
      Feynman28, Lecture I.29.4

      Arguments:
          omega: float or array-like, default range (1.0,10.0)
          c: float or array-like, default range (1.0,10.0)
      Returns:
          f: omega/c
      """
      return omega/c
  

class Feynman29:
  equation_lambda = lambda args : (lambda x1,x2,theta1,theta2: np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman29, Lecture I.29.16

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','theta1','theta2','x']
      """
      x1 = np.random.uniform(1.0,5.0, size)
      x2 = np.random.uniform(1.0,5.0, size)
      theta1 = np.random.uniform(1.0,5.0, size)
      theta2 = np.random.uniform(1.0,5.0, size)
      return Feynman29.calculate_df(x1,x2,theta1,theta2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x1,x2,theta1,theta2, noise_level = 0, include_original_target = False):
      """
      Feynman29, Lecture I.29.16

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          theta1: float or array-like, default range (1.0,5.0)
          theta2: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','x2','theta1','theta2','x']
      """
      target = Feynman29.calculate(x1,x2,theta1,theta2)
      data = [x1,x2,theta1,theta2]
      data.append(Noise(target,noise_level))
      columns = ['x1','x2','theta1','theta2','x']

      if(include_original_target):
         data.append(target)
         columns.append('x_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman29, Lecture I.29.16

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          theta1: float or array-like, default range (1.0,5.0)
          theta2: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
      """
      x1 = X[0]
      x2 = X[1]
      theta1 = X[2]
      theta2 = X[3]
      return Feynman29.calculate(x1,x2,theta1,theta2)

  @staticmethod
  def calculate(x1,x2,theta1,theta2):
      """
      Feynman29, Lecture I.29.16

      Arguments:
          x1: float or array-like, default range (1.0,5.0)
          x2: float or array-like, default range (1.0,5.0)
          theta1: float or array-like, default range (1.0,5.0)
          theta2: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
      """
      return np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))
  

class Feynman30:
  equation_lambda = lambda args : (lambda Int_0,theta,n: Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman30, Lecture I.30.3

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Int_0','theta','n','Int']
      """
      Int_0 = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      n = np.random.uniform(1.0,5.0, size)
      return Feynman30.calculate_df(Int_0,theta,n,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Int_0,theta,n, noise_level = 0, include_original_target = False):
      """
      Feynman30, Lecture I.30.3

      Arguments:
          Int_0: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Int_0','theta','n','Int']
      """
      target = Feynman30.calculate(Int_0,theta,n)
      data = [Int_0,theta,n]
      data.append(Noise(target,noise_level))
      columns = ['Int_0','theta','n','Int']

      if(include_original_target):
         data.append(target)
         columns.append('Int_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman30, Lecture I.30.3

      Arguments:
          Int_0: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: Int_0*sin(n*theta/2)**2/sin(theta/2)**2
      """
      Int_0 = X[0]
      theta = X[1]
      n = X[2]
      return Feynman30.calculate(Int_0,theta,n)

  @staticmethod
  def calculate(Int_0,theta,n):
      """
      Feynman30, Lecture I.30.3

      Arguments:
          Int_0: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: Int_0*sin(n*theta/2)**2/sin(theta/2)**2
      """
      return Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2
  

class Feynman31:
  equation_lambda = lambda args : (lambda lambd,d,n: np.arcsin(lambd/(n*d)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman31, Lecture I.30.5

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['lambd','d','n','theta']
      """
      lambd = np.random.uniform(1.0,2.0, size)
      d = np.random.uniform(2.0,5.0, size)
      n = np.random.uniform(1.0,5.0, size)
      return Feynman31.calculate_df(lambd,d,n,noise_level,include_original_target)

  @staticmethod
  def calculate_df(lambd,d,n, noise_level = 0, include_original_target = False):
      """
      Feynman31, Lecture I.30.5

      Arguments:
          lambd: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['lambd','d','n','theta']
      """
      target = Feynman31.calculate(lambd,d,n)
      data = [lambd,d,n]
      data.append(Noise(target,noise_level))
      columns = ['lambd','d','n','theta']

      if(include_original_target):
         data.append(target)
         columns.append('theta_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman31, Lecture I.30.5

      Arguments:
          lambd: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: arcsin(lambd/(n*d))
      """
      lambd = X[0]
      d = X[1]
      n = X[2]
      return Feynman31.calculate(lambd,d,n)

  @staticmethod
  def calculate(lambd,d,n):
      """
      Feynman31, Lecture I.30.5

      Arguments:
          lambd: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,5.0)
          n: float or array-like, default range (1.0,5.0)
      Returns:
          f: arcsin(lambd/(n*d))
      """
      return np.arcsin(lambd/(n*d))
  

class Feynman32:
  equation_lambda = lambda args : (lambda q,a,epsilon,c: q**2*a**2/(6*np.pi*epsilon*c**3) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman32, Lecture I.32.5

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','a','epsilon','c','Pwr']
      """
      q = np.random.uniform(1.0,5.0, size)
      a = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      return Feynman32.calculate_df(q,a,epsilon,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,a,epsilon,c, noise_level = 0, include_original_target = False):
      """
      Feynman32, Lecture I.32.5

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          a: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','a','epsilon','c','Pwr']
      """
      target = Feynman32.calculate(q,a,epsilon,c)
      data = [q,a,epsilon,c]
      data.append(Noise(target,noise_level))
      columns = ['q','a','epsilon','c','Pwr']

      if(include_original_target):
         data.append(target)
         columns.append('Pwr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman32, Lecture I.32.5

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          a: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: q**2*a**2/(6*pi*epsilon*c**3)
      """
      q = X[0]
      a = X[1]
      epsilon = X[2]
      c = X[3]
      return Feynman32.calculate(q,a,epsilon,c)

  @staticmethod
  def calculate(q,a,epsilon,c):
      """
      Feynman32, Lecture I.32.5

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          a: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: q**2*a**2/(6*pi*epsilon*c**3)
      """
      return q**2*a**2/(6*np.pi*epsilon*c**3)
  

class Feynman33:
  equation_lambda = lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman33, Lecture I.32.17

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','Ef','r','omega','omega_0','Pwr']
      """
      epsilon = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(1.0,2.0, size)
      Ef = np.random.uniform(1.0,2.0, size)
      r = np.random.uniform(1.0,2.0, size)
      omega = np.random.uniform(1.0,2.0, size)
      omega_0 = np.random.uniform(3.0,5.0, size)
      return Feynman33.calculate_df(epsilon,c,Ef,r,omega,omega_0,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,c,Ef,r,omega,omega_0, noise_level = 0, include_original_target = False):
      """
      Feynman33, Lecture I.32.17

      Arguments:
          epsilon: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
          r: float or array-like, default range (1.0,2.0)
          omega: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (3.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','Ef','r','omega','omega_0','Pwr']
      """
      target = Feynman33.calculate(epsilon,c,Ef,r,omega,omega_0)
      data = [epsilon,c,Ef,r,omega,omega_0]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','c','Ef','r','omega','omega_0','Pwr']

      if(include_original_target):
         data.append(target)
         columns.append('Pwr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman33, Lecture I.32.17

      Arguments:
          epsilon: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
          r: float or array-like, default range (1.0,2.0)
          omega: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (3.0,5.0)
      Returns:
          f: (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
      """
      epsilon = X[0]
      c = X[1]
      Ef = X[2]
      r = X[3]
      omega = X[4]
      omega_0 = X[5]
      return Feynman33.calculate(epsilon,c,Ef,r,omega,omega_0)

  @staticmethod
  def calculate(epsilon,c,Ef,r,omega,omega_0):
      """
      Feynman33, Lecture I.32.17

      Arguments:
          epsilon: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
          r: float or array-like, default range (1.0,2.0)
          omega: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (3.0,5.0)
      Returns:
          f: (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
      """
      return (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
  

class Feynman34:
  equation_lambda = lambda args : (lambda q,v,B,p: q*v*B/p )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman34, Lecture I.34.8

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','B','p','omega']
      """
      q = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      p = np.random.uniform(1.0,5.0, size)
      return Feynman34.calculate_df(q,v,B,p,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,v,B,p, noise_level = 0, include_original_target = False):
      """
      Feynman34, Lecture I.34.8

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','B','p','omega']
      """
      target = Feynman34.calculate(q,v,B,p)
      data = [q,v,B,p]
      data.append(Noise(target,noise_level))
      columns = ['q','v','B','p','omega']

      if(include_original_target):
         data.append(target)
         columns.append('omega_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman34, Lecture I.34.8

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v*B/p
      """
      q = X[0]
      v = X[1]
      B = X[2]
      p = X[3]
      return Feynman34.calculate(q,v,B,p)

  @staticmethod
  def calculate(q,v,B,p):
      """
      Feynman34, Lecture I.34.8

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v*B/p
      """
      return q*v*B/p
  

class Feynman35:
  equation_lambda = lambda args : (lambda c,v,omega_0: omega_0/(1-v/c) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman35, Lecture I.34.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega_0','omega']
      """
      c = np.random.uniform(3.0,10.0, size)
      v = np.random.uniform(1.0,2.0, size)
      omega_0 = np.random.uniform(1.0,5.0, size)
      return Feynman35.calculate_df(c,v,omega_0,noise_level,include_original_target)

  @staticmethod
  def calculate_df(c,v,omega_0, noise_level = 0, include_original_target = False):
      """
      Feynman35, Lecture I.34.1

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega_0','omega']
      """
      target = Feynman35.calculate(c,v,omega_0)
      data = [c,v,omega_0]
      data.append(Noise(target,noise_level))
      columns = ['c','v','omega_0','omega']

      if(include_original_target):
         data.append(target)
         columns.append('omega_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman35, Lecture I.34.1

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: omega_0/(1-v/c)
      """
      c = X[0]
      v = X[1]
      omega_0 = X[2]
      return Feynman35.calculate(c,v,omega_0)

  @staticmethod
  def calculate(c,v,omega_0):
      """
      Feynman35, Lecture I.34.1

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: omega_0/(1-v/c)
      """
      return omega_0/(1-v/c)
  

class Feynman36:
  equation_lambda = lambda args : (lambda c,v,omega_0: (1+v/c)/np.sqrt(1-v**2/c**2)*omega_0 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman36, Lecture I.34.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega_0','omega']
      """
      c = np.random.uniform(3.0,10.0, size)
      v = np.random.uniform(1.0,2.0, size)
      omega_0 = np.random.uniform(1.0,5.0, size)
      return Feynman36.calculate_df(c,v,omega_0,noise_level,include_original_target)

  @staticmethod
  def calculate_df(c,v,omega_0, noise_level = 0, include_original_target = False):
      """
      Feynman36, Lecture I.34.14

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega_0','omega']
      """
      target = Feynman36.calculate(c,v,omega_0)
      data = [c,v,omega_0]
      data.append(Noise(target,noise_level))
      columns = ['c','v','omega_0','omega']

      if(include_original_target):
         data.append(target)
         columns.append('omega_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman36, Lecture I.34.14

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: (1+v/c)/sqrt(1-v**2/c**2)*omega_0
      """
      c = X[0]
      v = X[1]
      omega_0 = X[2]
      return Feynman36.calculate(c,v,omega_0)

  @staticmethod
  def calculate(c,v,omega_0):
      """
      Feynman36, Lecture I.34.14

      Arguments:
          c: float or array-like, default range (3.0,10.0)
          v: float or array-like, default range (1.0,2.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: (1+v/c)/sqrt(1-v**2/c**2)*omega_0
      """
      return (1+v/c)/np.sqrt(1-v**2/c**2)*omega_0
  

class Feynman37:
  equation_lambda = lambda args : (lambda omega,h: (h/(2*np.pi))*omega )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman37, Lecture I.34.27

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','h','E_n']
      """
      omega = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      return Feynman37.calculate_df(omega,h,noise_level,include_original_target)

  @staticmethod
  def calculate_df(omega,h, noise_level = 0, include_original_target = False):
      """
      Feynman37, Lecture I.34.27

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','h','E_n']
      """
      target = Feynman37.calculate(omega,h)
      data = [omega,h]
      data.append(Noise(target,noise_level))
      columns = ['omega','h','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman37, Lecture I.34.27

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))*omega
      """
      omega = X[0]
      h = X[1]
      return Feynman37.calculate(omega,h)

  @staticmethod
  def calculate(omega,h):
      """
      Feynman37, Lecture I.34.27

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))*omega
      """
      return (h/(2*np.pi))*omega
  

class Feynman38:
  equation_lambda = lambda args : (lambda I1,I2,delta: I1+I2+2*np.sqrt(I1*I2)*np.cos(delta) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman38, Lecture I.37.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I1','I2','delta','Int']
      """
      I1 = np.random.uniform(1.0,5.0, size)
      I2 = np.random.uniform(1.0,5.0, size)
      delta = np.random.uniform(1.0,5.0, size)
      return Feynman38.calculate_df(I1,I2,delta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(I1,I2,delta, noise_level = 0, include_original_target = False):
      """
      Feynman38, Lecture I.37.4

      Arguments:
          I1: float or array-like, default range (1.0,5.0)
          I2: float or array-like, default range (1.0,5.0)
          delta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I1','I2','delta','Int']
      """
      target = Feynman38.calculate(I1,I2,delta)
      data = [I1,I2,delta]
      data.append(Noise(target,noise_level))
      columns = ['I1','I2','delta','Int']

      if(include_original_target):
         data.append(target)
         columns.append('Int_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman38, Lecture I.37.4

      Arguments:
          I1: float or array-like, default range (1.0,5.0)
          I2: float or array-like, default range (1.0,5.0)
          delta: float or array-like, default range (1.0,5.0)
      Returns:
          f: I1+I2+2*sqrt(I1*I2)*cos(delta)
      """
      I1 = X[0]
      I2 = X[1]
      delta = X[2]
      return Feynman38.calculate(I1,I2,delta)

  @staticmethod
  def calculate(I1,I2,delta):
      """
      Feynman38, Lecture I.37.4

      Arguments:
          I1: float or array-like, default range (1.0,5.0)
          I2: float or array-like, default range (1.0,5.0)
          delta: float or array-like, default range (1.0,5.0)
      Returns:
          f: I1+I2+2*sqrt(I1*I2)*cos(delta)
      """
      return I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)
  

class Feynman39:
  equation_lambda = lambda args : (lambda m,q,h,epsilon: 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman39, Lecture I.38.12

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','q','h','epsilon','r']
      """
      m = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      return Feynman39.calculate_df(m,q,h,epsilon,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,q,h,epsilon, noise_level = 0, include_original_target = False):
      """
      Feynman39, Lecture I.38.12

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','q','h','epsilon','r']
      """
      target = Feynman39.calculate(m,q,h,epsilon)
      data = [m,q,h,epsilon]
      data.append(Noise(target,noise_level))
      columns = ['m','q','h','epsilon','r']

      if(include_original_target):
         data.append(target)
         columns.append('r_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman39, Lecture I.38.12

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)
      """
      m = X[0]
      q = X[1]
      h = X[2]
      epsilon = X[3]
      return Feynman39.calculate(m,q,h,epsilon)

  @staticmethod
  def calculate(m,q,h,epsilon):
      """
      Feynman39, Lecture I.38.12

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)
      """
      return 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)
  

class Feynman40:
  equation_lambda = lambda args : (lambda pr,V: 3/2*pr*V )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman40, Lecture I.39.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['pr','V','E_n']
      """
      pr = np.random.uniform(1.0,5.0, size)
      V = np.random.uniform(1.0,5.0, size)
      return Feynman40.calculate_df(pr,V,noise_level,include_original_target)

  @staticmethod
  def calculate_df(pr,V, noise_level = 0, include_original_target = False):
      """
      Feynman40, Lecture I.39.1

      Arguments:
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['pr','V','E_n']
      """
      target = Feynman40.calculate(pr,V)
      data = [pr,V]
      data.append(Noise(target,noise_level))
      columns = ['pr','V','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman40, Lecture I.39.1

      Arguments:
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/2*pr*V
      """
      pr = X[0]
      V = X[1]
      return Feynman40.calculate(pr,V)

  @staticmethod
  def calculate(pr,V):
      """
      Feynman40, Lecture I.39.1

      Arguments:
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/2*pr*V
      """
      return 3/2*pr*V
  

class Feynman41:
  equation_lambda = lambda args : (lambda gamma,pr,V: 1/(gamma-1)*pr*V )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman41, Lecture I.39.11

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','pr','V','E_n']
      """
      gamma = np.random.uniform(2.0,5.0, size)
      pr = np.random.uniform(1.0,5.0, size)
      V = np.random.uniform(1.0,5.0, size)
      return Feynman41.calculate_df(gamma,pr,V,noise_level,include_original_target)

  @staticmethod
  def calculate_df(gamma,pr,V, noise_level = 0, include_original_target = False):
      """
      Feynman41, Lecture I.39.11

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','pr','V','E_n']
      """
      target = Feynman41.calculate(gamma,pr,V)
      data = [gamma,pr,V]
      data.append(Noise(target,noise_level))
      columns = ['gamma','pr','V','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman41, Lecture I.39.11

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(gamma-1)*pr*V
      """
      gamma = X[0]
      pr = X[1]
      V = X[2]
      return Feynman41.calculate(gamma,pr,V)

  @staticmethod
  def calculate(gamma,pr,V):
      """
      Feynman41, Lecture I.39.11

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(gamma-1)*pr*V
      """
      return 1/(gamma-1)*pr*V
  

class Feynman42:
  equation_lambda = lambda args : (lambda n,T,V,kb: n*kb*T/V )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman42, Lecture I.39.22

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','T','V','kb','pr']
      """
      n = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      V = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      return Feynman42.calculate_df(n,T,V,kb,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,T,V,kb, noise_level = 0, include_original_target = False):
      """
      Feynman42, Lecture I.39.22

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','T','V','kb','pr']
      """
      target = Feynman42.calculate(n,T,V,kb)
      data = [n,T,V,kb]
      data.append(Noise(target,noise_level))
      columns = ['n','T','V','kb','pr']

      if(include_original_target):
         data.append(target)
         columns.append('pr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman42, Lecture I.39.22

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*kb*T/V
      """
      n = X[0]
      T = X[1]
      V = X[2]
      kb = X[3]
      return Feynman42.calculate(n,T,V,kb)

  @staticmethod
  def calculate(n,T,V,kb):
      """
      Feynman42, Lecture I.39.22

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*kb*T/V
      """
      return n*kb*T/V
  

class Feynman43:
  equation_lambda = lambda args : (lambda n_0,m,x,T,g,kb: n_0*np.exp(-m*g*x/(kb*T)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman43, Lecture I.40.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','m','x','T','g','kb','n']
      """
      n_0 = np.random.uniform(1.0,5.0, size)
      m = np.random.uniform(1.0,5.0, size)
      x = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      g = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      return Feynman43.calculate_df(n_0,m,x,T,g,kb,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n_0,m,x,T,g,kb, noise_level = 0, include_original_target = False):
      """
      Feynman43, Lecture I.40.1

      Arguments:
          n_0: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','m','x','T','g','kb','n']
      """
      target = Feynman43.calculate(n_0,m,x,T,g,kb)
      data = [n_0,m,x,T,g,kb]
      data.append(Noise(target,noise_level))
      columns = ['n_0','m','x','T','g','kb','n']

      if(include_original_target):
         data.append(target)
         columns.append('n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman43, Lecture I.40.1

      Arguments:
          n_0: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_0*exp(-m*g*x/(kb*T))
      """
      n_0 = X[0]
      m = X[1]
      x = X[2]
      T = X[3]
      g = X[4]
      kb = X[5]
      return Feynman43.calculate(n_0,m,x,T,g,kb)

  @staticmethod
  def calculate(n_0,m,x,T,g,kb):
      """
      Feynman43, Lecture I.40.1

      Arguments:
          n_0: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          g: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_0*exp(-m*g*x/(kb*T))
      """
      return n_0*np.exp(-m*g*x/(kb*T))
  

class Feynman44:
  equation_lambda = lambda args : (lambda omega,T,h,kb,c: h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman44, Lecture I.41.16

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','T','h','kb','c','L_rad']
      """
      omega = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      return Feynman44.calculate_df(omega,T,h,kb,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(omega,T,h,kb,c, noise_level = 0, include_original_target = False):
      """
      Feynman44, Lecture I.41.16

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','T','h','kb','c','L_rad']
      """
      target = Feynman44.calculate(omega,T,h,kb,c)
      data = [omega,T,h,kb,c]
      data.append(Noise(target,noise_level))
      columns = ['omega','T','h','kb','c','L_rad']

      if(include_original_target):
         data.append(target)
         columns.append('L_rad_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman44, Lecture I.41.16

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))
      """
      omega = X[0]
      T = X[1]
      h = X[2]
      kb = X[3]
      c = X[4]
      return Feynman44.calculate(omega,T,h,kb,c)

  @staticmethod
  def calculate(omega,T,h,kb,c):
      """
      Feynman44, Lecture I.41.16

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))
      """
      return h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))
  

class Feynman45:
  equation_lambda = lambda args : (lambda mu_drift,q,Volt,d: mu_drift*q*Volt/d )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman45, Lecture I.43.16

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mu_drift','q','Volt','d','v']
      """
      mu_drift = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      Volt = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman45.calculate_df(mu_drift,q,Volt,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mu_drift,q,Volt,d, noise_level = 0, include_original_target = False):
      """
      Feynman45, Lecture I.43.16

      Arguments:
          mu_drift: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mu_drift','q','Volt','d','v']
      """
      target = Feynman45.calculate(mu_drift,q,Volt,d)
      data = [mu_drift,q,Volt,d]
      data.append(Noise(target,noise_level))
      columns = ['mu_drift','q','Volt','d','v']

      if(include_original_target):
         data.append(target)
         columns.append('v_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman45, Lecture I.43.16

      Arguments:
          mu_drift: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: mu_drift*q*Volt/d
      """
      mu_drift = X[0]
      q = X[1]
      Volt = X[2]
      d = X[3]
      return Feynman45.calculate(mu_drift,q,Volt,d)

  @staticmethod
  def calculate(mu_drift,q,Volt,d):
      """
      Feynman45, Lecture I.43.16

      Arguments:
          mu_drift: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: mu_drift*q*Volt/d
      """
      return mu_drift*q*Volt/d
  

class Feynman46:
  equation_lambda = lambda args : (lambda mob,T,kb: mob*kb*T )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman46, Lecture I.43.31

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mob','T','kb','D']
      """
      mob = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      return Feynman46.calculate_df(mob,T,kb,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mob,T,kb, noise_level = 0, include_original_target = False):
      """
      Feynman46, Lecture I.43.31

      Arguments:
          mob: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mob','T','kb','D']
      """
      target = Feynman46.calculate(mob,T,kb)
      data = [mob,T,kb]
      data.append(Noise(target,noise_level))
      columns = ['mob','T','kb','D']

      if(include_original_target):
         data.append(target)
         columns.append('D_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman46, Lecture I.43.31

      Arguments:
          mob: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: mob*kb*T
      """
      mob = X[0]
      T = X[1]
      kb = X[2]
      return Feynman46.calculate(mob,T,kb)

  @staticmethod
  def calculate(mob,T,kb):
      """
      Feynman46, Lecture I.43.31

      Arguments:
          mob: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
      Returns:
          f: mob*kb*T
      """
      return mob*kb*T
  

class Feynman47:
  equation_lambda = lambda args : (lambda gamma,kb,A,v: 1/(gamma-1)*kb*v/A )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman47, Lecture I.43.43

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','kb','A','v','kappa']
      """
      gamma = np.random.uniform(2.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      A = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      return Feynman47.calculate_df(gamma,kb,A,v,noise_level,include_original_target)

  @staticmethod
  def calculate_df(gamma,kb,A,v, noise_level = 0, include_original_target = False):
      """
      Feynman47, Lecture I.43.43

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','kb','A','v','kappa']
      """
      target = Feynman47.calculate(gamma,kb,A,v)
      data = [gamma,kb,A,v]
      data.append(Noise(target,noise_level))
      columns = ['gamma','kb','A','v','kappa']

      if(include_original_target):
         data.append(target)
         columns.append('kappa_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman47, Lecture I.43.43

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(gamma-1)*kb*v/A
      """
      gamma = X[0]
      kb = X[1]
      A = X[2]
      v = X[3]
      return Feynman47.calculate(gamma,kb,A,v)

  @staticmethod
  def calculate(gamma,kb,A,v):
      """
      Feynman47, Lecture I.43.43

      Arguments:
          gamma: float or array-like, default range (2.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(gamma-1)*kb*v/A
      """
      return 1/(gamma-1)*kb*v/A
  

class Feynman48:
  equation_lambda = lambda args : (lambda n,kb,T,V1,V2: n*kb*T*np.ln(V2/V1) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman48, Lecture I.44.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','kb','T','V1','V2','E_n']
      """
      n = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      V1 = np.random.uniform(1.0,5.0, size)
      V2 = np.random.uniform(1.0,5.0, size)
      return Feynman48.calculate_df(n,kb,T,V1,V2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,kb,T,V1,V2, noise_level = 0, include_original_target = False):
      """
      Feynman48, Lecture I.44.4

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V1: float or array-like, default range (1.0,5.0)
          V2: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','kb','T','V1','V2','E_n']
      """
      target = Feynman48.calculate(n,kb,T,V1,V2)
      data = [n,kb,T,V1,V2]
      data.append(Noise(target,noise_level))
      columns = ['n','kb','T','V1','V2','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman48, Lecture I.44.4

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V1: float or array-like, default range (1.0,5.0)
          V2: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*kb*T*ln(V2/V1)
      """
      n = X[0]
      kb = X[1]
      T = X[2]
      V1 = X[3]
      V2 = X[4]
      return Feynman48.calculate(n,kb,T,V1,V2)

  @staticmethod
  def calculate(n,kb,T,V1,V2):
      """
      Feynman48, Lecture I.44.4

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          V1: float or array-like, default range (1.0,5.0)
          V2: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*kb*T*ln(V2/V1)
      """
      return n*kb*T*np.ln(V2/V1)
  

class Feynman49:
  equation_lambda = lambda args : (lambda gamma,pr,rho: np.sqrt(gamma*pr/rho) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman49, Lecture I.47.23

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','pr','rho','c']
      """
      gamma = np.random.uniform(1.0,5.0, size)
      pr = np.random.uniform(1.0,5.0, size)
      rho = np.random.uniform(1.0,5.0, size)
      return Feynman49.calculate_df(gamma,pr,rho,noise_level,include_original_target)

  @staticmethod
  def calculate_df(gamma,pr,rho, noise_level = 0, include_original_target = False):
      """
      Feynman49, Lecture I.47.23

      Arguments:
          gamma: float or array-like, default range (1.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          rho: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['gamma','pr','rho','c']
      """
      target = Feynman49.calculate(gamma,pr,rho)
      data = [gamma,pr,rho]
      data.append(Noise(target,noise_level))
      columns = ['gamma','pr','rho','c']

      if(include_original_target):
         data.append(target)
         columns.append('c_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman49, Lecture I.47.23

      Arguments:
          gamma: float or array-like, default range (1.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          rho: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt(gamma*pr/rho)
      """
      gamma = X[0]
      pr = X[1]
      rho = X[2]
      return Feynman49.calculate(gamma,pr,rho)

  @staticmethod
  def calculate(gamma,pr,rho):
      """
      Feynman49, Lecture I.47.23

      Arguments:
          gamma: float or array-like, default range (1.0,5.0)
          pr: float or array-like, default range (1.0,5.0)
          rho: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt(gamma*pr/rho)
      """
      return np.sqrt(gamma*pr/rho)
  

class Feynman50:
  equation_lambda = lambda args : (lambda m,v,c: m*c**2/np.sqrt(1-v**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman50, Lecture I.48.2

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','v','c','E_n']
      """
      m = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman50.calculate_df(m,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman50, Lecture I.48.2

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','v','c','E_n']
      """
      target = Feynman50.calculate(m,v,c)
      data = [m,v,c]
      data.append(Noise(target,noise_level))
      columns = ['m','v','c','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman50, Lecture I.48.2

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m*c**2/sqrt(1-v**2/c**2)
      """
      m = X[0]
      v = X[1]
      c = X[2]
      return Feynman50.calculate(m,v,c)

  @staticmethod
  def calculate(m,v,c):
      """
      Feynman50, Lecture I.48.2

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: m*c**2/sqrt(1-v**2/c**2)
      """
      return m*c**2/np.sqrt(1-v**2/c**2)
  

class Feynman51:
  equation_lambda = lambda args : (lambda x1,omega,t,alpha: x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman51, Lecture I.50.26

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','omega','t','alpha','x']
      """
      x1 = np.random.uniform(1.0,3.0, size)
      omega = np.random.uniform(1.0,3.0, size)
      t = np.random.uniform(1.0,3.0, size)
      alpha = np.random.uniform(1.0,3.0, size)
      return Feynman51.calculate_df(x1,omega,t,alpha,noise_level,include_original_target)

  @staticmethod
  def calculate_df(x1,omega,t,alpha, noise_level = 0, include_original_target = False):
      """
      Feynman51, Lecture I.50.26

      Arguments:
          x1: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['x1','omega','t','alpha','x']
      """
      target = Feynman51.calculate(x1,omega,t,alpha)
      data = [x1,omega,t,alpha]
      data.append(Noise(target,noise_level))
      columns = ['x1','omega','t','alpha','x']

      if(include_original_target):
         data.append(target)
         columns.append('x_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman51, Lecture I.50.26

      Arguments:
          x1: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
      Returns:
          f: x1*(cos(omega*t)+alpha*cos(omega*t)**2)
      """
      x1 = X[0]
      omega = X[1]
      t = X[2]
      alpha = X[3]
      return Feynman51.calculate(x1,omega,t,alpha)

  @staticmethod
  def calculate(x1,omega,t,alpha):
      """
      Feynman51, Lecture I.50.26

      Arguments:
          x1: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
      Returns:
          f: x1*(cos(omega*t)+alpha*cos(omega*t)**2)
      """
      return x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)
  

class Feynman52:
  equation_lambda = lambda args : (lambda kappa,T1,T2,A,d: kappa*(T2-T1)*A/d )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman52, Lecture II.2.42

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['kappa','T1','T2','A','d','Pwr']
      """
      kappa = np.random.uniform(1.0,5.0, size)
      T1 = np.random.uniform(1.0,5.0, size)
      T2 = np.random.uniform(1.0,5.0, size)
      A = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman52.calculate_df(kappa,T1,T2,A,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(kappa,T1,T2,A,d, noise_level = 0, include_original_target = False):
      """
      Feynman52, Lecture II.2.42

      Arguments:
          kappa: float or array-like, default range (1.0,5.0)
          T1: float or array-like, default range (1.0,5.0)
          T2: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['kappa','T1','T2','A','d','Pwr']
      """
      target = Feynman52.calculate(kappa,T1,T2,A,d)
      data = [kappa,T1,T2,A,d]
      data.append(Noise(target,noise_level))
      columns = ['kappa','T1','T2','A','d','Pwr']

      if(include_original_target):
         data.append(target)
         columns.append('Pwr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman52, Lecture II.2.42

      Arguments:
          kappa: float or array-like, default range (1.0,5.0)
          T1: float or array-like, default range (1.0,5.0)
          T2: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: kappa*(T2-T1)*A/d
      """
      kappa = X[0]
      T1 = X[1]
      T2 = X[2]
      A = X[3]
      d = X[4]
      return Feynman52.calculate(kappa,T1,T2,A,d)

  @staticmethod
  def calculate(kappa,T1,T2,A,d):
      """
      Feynman52, Lecture II.2.42

      Arguments:
          kappa: float or array-like, default range (1.0,5.0)
          T1: float or array-like, default range (1.0,5.0)
          T2: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: kappa*(T2-T1)*A/d
      """
      return kappa*(T2-T1)*A/d
  

class Feynman53:
  equation_lambda = lambda args : (lambda Pwr,r: Pwr/(4*np.pi*r**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman53, Lecture II.3.24

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Pwr','r','flux']
      """
      Pwr = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman53.calculate_df(Pwr,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Pwr,r, noise_level = 0, include_original_target = False):
      """
      Feynman53, Lecture II.3.24

      Arguments:
          Pwr: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Pwr','r','flux']
      """
      target = Feynman53.calculate(Pwr,r)
      data = [Pwr,r]
      data.append(Noise(target,noise_level))
      columns = ['Pwr','r','flux']

      if(include_original_target):
         data.append(target)
         columns.append('flux_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman53, Lecture II.3.24

      Arguments:
          Pwr: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: Pwr/(4*pi*r**2)
      """
      Pwr = X[0]
      r = X[1]
      return Feynman53.calculate(Pwr,r)

  @staticmethod
  def calculate(Pwr,r):
      """
      Feynman53, Lecture II.3.24

      Arguments:
          Pwr: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: Pwr/(4*pi*r**2)
      """
      return Pwr/(4*np.pi*r**2)
  

class Feynman54:
  equation_lambda = lambda args : (lambda q,epsilon,r: q/(4*np.pi*epsilon*r) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman54, Lecture II.4.23

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','r','Volt']
      """
      q = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman54.calculate_df(q,epsilon,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,epsilon,r, noise_level = 0, include_original_target = False):
      """
      Feynman54, Lecture II.4.23

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','r','Volt']
      """
      target = Feynman54.calculate(q,epsilon,r)
      data = [q,epsilon,r]
      data.append(Noise(target,noise_level))
      columns = ['q','epsilon','r','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman54, Lecture II.4.23

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/(4*pi*epsilon*r)
      """
      q = X[0]
      epsilon = X[1]
      r = X[2]
      return Feynman54.calculate(q,epsilon,r)

  @staticmethod
  def calculate(q,epsilon,r):
      """
      Feynman54, Lecture II.4.23

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/(4*pi*epsilon*r)
      """
      return q/(4*np.pi*epsilon*r)
  

class Feynman55:
  equation_lambda = lambda args : (lambda epsilon,p_d,theta,r: 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman55, Lecture II.6.11

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','theta','r','Volt']
      """
      epsilon = np.random.uniform(1.0,3.0, size)
      p_d = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      r = np.random.uniform(1.0,3.0, size)
      return Feynman55.calculate_df(epsilon,p_d,theta,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,p_d,theta,r, noise_level = 0, include_original_target = False):
      """
      Feynman55, Lecture II.6.11

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','theta','r','Volt']
      """
      target = Feynman55.calculate(epsilon,p_d,theta,r)
      data = [epsilon,p_d,theta,r]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','p_d','theta','r','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman55, Lecture II.6.11

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: 1/(4*pi*epsilon)*p_d*cos(theta)/r**2
      """
      epsilon = X[0]
      p_d = X[1]
      theta = X[2]
      r = X[3]
      return Feynman55.calculate(epsilon,p_d,theta,r)

  @staticmethod
  def calculate(epsilon,p_d,theta,r):
      """
      Feynman55, Lecture II.6.11

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: 1/(4*pi*epsilon)*p_d*cos(theta)/r**2
      """
      return 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2
  

class Feynman56:
  equation_lambda = lambda args : (lambda epsilon,p_d,r,x,y,z: p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman56, Lecture II.6.15a

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','r','x','y','z','Ef']
      """
      epsilon = np.random.uniform(1.0,3.0, size)
      p_d = np.random.uniform(1.0,3.0, size)
      r = np.random.uniform(1.0,3.0, size)
      x = np.random.uniform(1.0,3.0, size)
      y = np.random.uniform(1.0,3.0, size)
      z = np.random.uniform(1.0,3.0, size)
      return Feynman56.calculate_df(epsilon,p_d,r,x,y,z,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,p_d,r,x,y,z, noise_level = 0, include_original_target = False):
      """
      Feynman56, Lecture II.6.15a

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
          y: float or array-like, default range (1.0,3.0)
          z: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','r','x','y','z','Ef']
      """
      target = Feynman56.calculate(epsilon,p_d,r,x,y,z)
      data = [epsilon,p_d,r,x,y,z]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','p_d','r','x','y','z','Ef']

      if(include_original_target):
         data.append(target)
         columns.append('Ef_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman56, Lecture II.6.15a

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
          y: float or array-like, default range (1.0,3.0)
          z: float or array-like, default range (1.0,3.0)
      Returns:
          f: p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)
      """
      epsilon = X[0]
      p_d = X[1]
      r = X[2]
      x = X[3]
      y = X[4]
      z = X[5]
      return Feynman56.calculate(epsilon,p_d,r,x,y,z)

  @staticmethod
  def calculate(epsilon,p_d,r,x,y,z):
      """
      Feynman56, Lecture II.6.15a

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          x: float or array-like, default range (1.0,3.0)
          y: float or array-like, default range (1.0,3.0)
          z: float or array-like, default range (1.0,3.0)
      Returns:
          f: p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)
      """
      return p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)
  

class Feynman57:
  equation_lambda = lambda args : (lambda epsilon,p_d,theta,r: p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman57, Lecture II.6.15b

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','theta','r','Ef']
      """
      epsilon = np.random.uniform(1.0,3.0, size)
      p_d = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      r = np.random.uniform(1.0,3.0, size)
      return Feynman57.calculate_df(epsilon,p_d,theta,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,p_d,theta,r, noise_level = 0, include_original_target = False):
      """
      Feynman57, Lecture II.6.15b

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','p_d','theta','r','Ef']
      """
      target = Feynman57.calculate(epsilon,p_d,theta,r)
      data = [epsilon,p_d,theta,r]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','p_d','theta','r','Ef']

      if(include_original_target):
         data.append(target)
         columns.append('Ef_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman57, Lecture II.6.15b

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3
      """
      epsilon = X[0]
      p_d = X[1]
      theta = X[2]
      r = X[3]
      return Feynman57.calculate(epsilon,p_d,theta,r)

  @staticmethod
  def calculate(epsilon,p_d,theta,r):
      """
      Feynman57, Lecture II.6.15b

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3
      """
      return p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3
  

class Feynman58:
  equation_lambda = lambda args : (lambda q,epsilon,d: 3/5*q**2/(4*np.pi*epsilon*d) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman58, Lecture II.8.7

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','d','E_n']
      """
      q = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman58.calculate_df(q,epsilon,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,epsilon,d, noise_level = 0, include_original_target = False):
      """
      Feynman58, Lecture II.8.7

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','d','E_n']
      """
      target = Feynman58.calculate(q,epsilon,d)
      data = [q,epsilon,d]
      data.append(Noise(target,noise_level))
      columns = ['q','epsilon','d','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman58, Lecture II.8.7

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/5*q**2/(4*pi*epsilon*d)
      """
      q = X[0]
      epsilon = X[1]
      d = X[2]
      return Feynman58.calculate(q,epsilon,d)

  @staticmethod
  def calculate(q,epsilon,d):
      """
      Feynman58, Lecture II.8.7

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/5*q**2/(4*pi*epsilon*d)
      """
      return 3/5*q**2/(4*np.pi*epsilon*d)
  

class Feynman59:
  equation_lambda = lambda args : (lambda epsilon,Ef: epsilon*Ef**2/2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman59, Lecture II.8.31

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','Ef','E_den']
      """
      epsilon = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      return Feynman59.calculate_df(epsilon,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman59, Lecture II.8.31

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','Ef','E_den']
      """
      target = Feynman59.calculate(epsilon,Ef)
      data = [epsilon,Ef]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','Ef','E_den']

      if(include_original_target):
         data.append(target)
         columns.append('E_den_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman59, Lecture II.8.31

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*Ef**2/2
      """
      epsilon = X[0]
      Ef = X[1]
      return Feynman59.calculate(epsilon,Ef)

  @staticmethod
  def calculate(epsilon,Ef):
      """
      Feynman59, Lecture II.8.31

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*Ef**2/2
      """
      return epsilon*Ef**2/2
  

class Feynman60:
  equation_lambda = lambda args : (lambda sigma_den,epsilon,chi: sigma_den/epsilon*1/(1+chi) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman60, Lecture II.10.9

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma_den','epsilon','chi','Ef']
      """
      sigma_den = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      chi = np.random.uniform(1.0,5.0, size)
      return Feynman60.calculate_df(sigma_den,epsilon,chi,noise_level,include_original_target)

  @staticmethod
  def calculate_df(sigma_den,epsilon,chi, noise_level = 0, include_original_target = False):
      """
      Feynman60, Lecture II.10.9

      Arguments:
          sigma_den: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['sigma_den','epsilon','chi','Ef']
      """
      target = Feynman60.calculate(sigma_den,epsilon,chi)
      data = [sigma_den,epsilon,chi]
      data.append(Noise(target,noise_level))
      columns = ['sigma_den','epsilon','chi','Ef']

      if(include_original_target):
         data.append(target)
         columns.append('Ef_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman60, Lecture II.10.9

      Arguments:
          sigma_den: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
      Returns:
          f: sigma_den/epsilon*1/(1+chi)
      """
      sigma_den = X[0]
      epsilon = X[1]
      chi = X[2]
      return Feynman60.calculate(sigma_den,epsilon,chi)

  @staticmethod
  def calculate(sigma_den,epsilon,chi):
      """
      Feynman60, Lecture II.10.9

      Arguments:
          sigma_den: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
      Returns:
          f: sigma_den/epsilon*1/(1+chi)
      """
      return sigma_den/epsilon*1/(1+chi)
  

class Feynman61:
  equation_lambda = lambda args : (lambda q,Ef,m,omega_0,omega: q*Ef/(m*(omega_0**2-omega**2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman61, Lecture II.11.3

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','Ef','m','omega_0','omega','x']
      """
      q = np.random.uniform(1.0,3.0, size)
      Ef = np.random.uniform(1.0,3.0, size)
      m = np.random.uniform(1.0,3.0, size)
      omega_0 = np.random.uniform(3.0,5.0, size)
      omega = np.random.uniform(1.0,2.0, size)
      return Feynman61.calculate_df(q,Ef,m,omega_0,omega,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,Ef,m,omega_0,omega, noise_level = 0, include_original_target = False):
      """
      Feynman61, Lecture II.11.3

      Arguments:
          q: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (3.0,5.0)
          omega: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','Ef','m','omega_0','omega','x']
      """
      target = Feynman61.calculate(q,Ef,m,omega_0,omega)
      data = [q,Ef,m,omega_0,omega]
      data.append(Noise(target,noise_level))
      columns = ['q','Ef','m','omega_0','omega','x']

      if(include_original_target):
         data.append(target)
         columns.append('x_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman61, Lecture II.11.3

      Arguments:
          q: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (3.0,5.0)
          omega: float or array-like, default range (1.0,2.0)
      Returns:
          f: q*Ef/(m*(omega_0**2-omega**2))
      """
      q = X[0]
      Ef = X[1]
      m = X[2]
      omega_0 = X[3]
      omega = X[4]
      return Feynman61.calculate(q,Ef,m,omega_0,omega)

  @staticmethod
  def calculate(q,Ef,m,omega_0,omega):
      """
      Feynman61, Lecture II.11.3

      Arguments:
          q: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          omega_0: float or array-like, default range (3.0,5.0)
          omega: float or array-like, default range (1.0,2.0)
      Returns:
          f: q*Ef/(m*(omega_0**2-omega**2))
      """
      return q*Ef/(m*(omega_0**2-omega**2))
  

class Feynman62:
  equation_lambda = lambda args : (lambda n_0,kb,T,theta,p_d,Ef: n_0*(1+p_d*Ef*np.cos(theta)/(kb*T)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman62, Lecture II.11.17

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','kb','T','theta','p_d','Ef','n']
      """
      n_0 = np.random.uniform(1.0,3.0, size)
      kb = np.random.uniform(1.0,3.0, size)
      T = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      p_d = np.random.uniform(1.0,3.0, size)
      Ef = np.random.uniform(1.0,3.0, size)
      return Feynman62.calculate_df(n_0,kb,T,theta,p_d,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n_0,kb,T,theta,p_d,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman62, Lecture II.11.17

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','kb','T','theta','p_d','Ef','n']
      """
      target = Feynman62.calculate(n_0,kb,T,theta,p_d,Ef)
      data = [n_0,kb,T,theta,p_d,Ef]
      data.append(Noise(target,noise_level))
      columns = ['n_0','kb','T','theta','p_d','Ef','n']

      if(include_original_target):
         data.append(target)
         columns.append('n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman62, Lecture II.11.17

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
      Returns:
          f: n_0*(1+p_d*Ef*cos(theta)/(kb*T))
      """
      n_0 = X[0]
      kb = X[1]
      T = X[2]
      theta = X[3]
      p_d = X[4]
      Ef = X[5]
      return Feynman62.calculate(n_0,kb,T,theta,p_d,Ef)

  @staticmethod
  def calculate(n_0,kb,T,theta,p_d,Ef):
      """
      Feynman62, Lecture II.11.17

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
      Returns:
          f: n_0*(1+p_d*Ef*cos(theta)/(kb*T))
      """
      return n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))
  

class Feynman63:
  equation_lambda = lambda args : (lambda n_rho,p_d,Ef,kb,T: n_rho*p_d**2*Ef/(3*kb*T) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman63, Lecture II.11.20

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_rho','p_d','Ef','kb','T','Pol']
      """
      n_rho = np.random.uniform(1.0,5.0, size)
      p_d = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      return Feynman63.calculate_df(n_rho,p_d,Ef,kb,T,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n_rho,p_d,Ef,kb,T, noise_level = 0, include_original_target = False):
      """
      Feynman63, Lecture II.11.20

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_rho','p_d','Ef','kb','T','Pol']
      """
      target = Feynman63.calculate(n_rho,p_d,Ef,kb,T)
      data = [n_rho,p_d,Ef,kb,T]
      data.append(Noise(target,noise_level))
      columns = ['n_rho','p_d','Ef','kb','T','Pol']

      if(include_original_target):
         data.append(target)
         columns.append('Pol_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman63, Lecture II.11.20

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_rho*p_d**2*Ef/(3*kb*T)
      """
      n_rho = X[0]
      p_d = X[1]
      Ef = X[2]
      kb = X[3]
      T = X[4]
      return Feynman63.calculate(n_rho,p_d,Ef,kb,T)

  @staticmethod
  def calculate(n_rho,p_d,Ef,kb,T):
      """
      Feynman63, Lecture II.11.20

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_rho*p_d**2*Ef/(3*kb*T)
      """
      return n_rho*p_d**2*Ef/(3*kb*T)
  

class Feynman64:
  equation_lambda = lambda args : (lambda n,alpha,epsilon,Ef: n*alpha/(1-(n*alpha/3))*epsilon*Ef )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman64, Lecture II.11.27

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','alpha','epsilon','Ef','Pol']
      """
      n = np.random.uniform(0.0,1.0, size)
      alpha = np.random.uniform(0.0,1.0, size)
      epsilon = np.random.uniform(1.0,2.0, size)
      Ef = np.random.uniform(1.0,2.0, size)
      return Feynman64.calculate_df(n,alpha,epsilon,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,alpha,epsilon,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman64, Lecture II.11.27

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
          epsilon: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','alpha','epsilon','Ef','Pol']
      """
      target = Feynman64.calculate(n,alpha,epsilon,Ef)
      data = [n,alpha,epsilon,Ef]
      data.append(Noise(target,noise_level))
      columns = ['n','alpha','epsilon','Ef','Pol']

      if(include_original_target):
         data.append(target)
         columns.append('Pol_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman64, Lecture II.11.27

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
          epsilon: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
      Returns:
          f: n*alpha/(1-(n*alpha/3))*epsilon*Ef
      """
      n = X[0]
      alpha = X[1]
      epsilon = X[2]
      Ef = X[3]
      return Feynman64.calculate(n,alpha,epsilon,Ef)

  @staticmethod
  def calculate(n,alpha,epsilon,Ef):
      """
      Feynman64, Lecture II.11.27

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
          epsilon: float or array-like, default range (1.0,2.0)
          Ef: float or array-like, default range (1.0,2.0)
      Returns:
          f: n*alpha/(1-(n*alpha/3))*epsilon*Ef
      """
      return n*alpha/(1-(n*alpha/3))*epsilon*Ef
  

class Feynman65:
  equation_lambda = lambda args : (lambda n,alpha: 1+n*alpha/(1-(n*alpha/3)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman65, Lecture II.11.28

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','alpha','theta']
      """
      n = np.random.uniform(0.0,1.0, size)
      alpha = np.random.uniform(0.0,1.0, size)
      return Feynman65.calculate_df(n,alpha,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,alpha, noise_level = 0, include_original_target = False):
      """
      Feynman65, Lecture II.11.28

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','alpha','theta']
      """
      target = Feynman65.calculate(n,alpha)
      data = [n,alpha]
      data.append(Noise(target,noise_level))
      columns = ['n','alpha','theta']

      if(include_original_target):
         data.append(target)
         columns.append('theta_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman65, Lecture II.11.28

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
      Returns:
          f: 1+n*alpha/(1-(n*alpha/3))
      """
      n = X[0]
      alpha = X[1]
      return Feynman65.calculate(n,alpha)

  @staticmethod
  def calculate(n,alpha):
      """
      Feynman65, Lecture II.11.28

      Arguments:
          n: float or array-like, default range (0.0,1.0)
          alpha: float or array-like, default range (0.0,1.0)
      Returns:
          f: 1+n*alpha/(1-(n*alpha/3))
      """
      return 1+n*alpha/(1-(n*alpha/3))
  

class Feynman66:
  equation_lambda = lambda args : (lambda epsilon,c,I,r: 1/(4*np.pi*epsilon*c**2)*2*I/r )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman66, Lecture II.13.17

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','I','r','B']
      """
      epsilon = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      I = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman66.calculate_df(epsilon,c,I,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,c,I,r, noise_level = 0, include_original_target = False):
      """
      Feynman66, Lecture II.13.17

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          I: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','I','r','B']
      """
      target = Feynman66.calculate(epsilon,c,I,r)
      data = [epsilon,c,I,r]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','c','I','r','B']

      if(include_original_target):
         data.append(target)
         columns.append('B_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman66, Lecture II.13.17

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          I: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(4*pi*epsilon*c**2)*2*I/r
      """
      epsilon = X[0]
      c = X[1]
      I = X[2]
      r = X[3]
      return Feynman66.calculate(epsilon,c,I,r)

  @staticmethod
  def calculate(epsilon,c,I,r):
      """
      Feynman66, Lecture II.13.17

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          I: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(4*pi*epsilon*c**2)*2*I/r
      """
      return 1/(4*np.pi*epsilon*c**2)*2*I/r
  

class Feynman67:
  equation_lambda = lambda args : (lambda rho_c_0,v,c: rho_c_0/np.sqrt(1-v**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman67, Lecture II.13.23

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','v','c','rho_c']
      """
      rho_c_0 = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman67.calculate_df(rho_c_0,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(rho_c_0,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman67, Lecture II.13.23

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','v','c','rho_c']
      """
      target = Feynman67.calculate(rho_c_0,v,c)
      data = [rho_c_0,v,c]
      data.append(Noise(target,noise_level))
      columns = ['rho_c_0','v','c','rho_c']

      if(include_original_target):
         data.append(target)
         columns.append('rho_c_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman67, Lecture II.13.23

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: rho_c_0/sqrt(1-v**2/c**2)
      """
      rho_c_0 = X[0]
      v = X[1]
      c = X[2]
      return Feynman67.calculate(rho_c_0,v,c)

  @staticmethod
  def calculate(rho_c_0,v,c):
      """
      Feynman67, Lecture II.13.23

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: rho_c_0/sqrt(1-v**2/c**2)
      """
      return rho_c_0/np.sqrt(1-v**2/c**2)
  

class Feynman68:
  equation_lambda = lambda args : (lambda rho_c_0,v,c: rho_c_0*v/np.sqrt(1-v**2/c**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman68, Lecture II.13.34

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','v','c','j']
      """
      rho_c_0 = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman68.calculate_df(rho_c_0,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(rho_c_0,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman68, Lecture II.13.34

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','v','c','j']
      """
      target = Feynman68.calculate(rho_c_0,v,c)
      data = [rho_c_0,v,c]
      data.append(Noise(target,noise_level))
      columns = ['rho_c_0','v','c','j']

      if(include_original_target):
         data.append(target)
         columns.append('j_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman68, Lecture II.13.34

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: rho_c_0*v/sqrt(1-v**2/c**2)
      """
      rho_c_0 = X[0]
      v = X[1]
      c = X[2]
      return Feynman68.calculate(rho_c_0,v,c)

  @staticmethod
  def calculate(rho_c_0,v,c):
      """
      Feynman68, Lecture II.13.34

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: rho_c_0*v/sqrt(1-v**2/c**2)
      """
      return rho_c_0*v/np.sqrt(1-v**2/c**2)
  

class Feynman69:
  equation_lambda = lambda args : (lambda mom,B,theta: -mom*B*np.cos(theta) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman69, Lecture II.15.4

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','theta','E_n']
      """
      mom = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      return Feynman69.calculate_df(mom,B,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mom,B,theta, noise_level = 0, include_original_target = False):
      """
      Feynman69, Lecture II.15.4

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','theta','E_n']
      """
      target = Feynman69.calculate(mom,B,theta)
      data = [mom,B,theta]
      data.append(Noise(target,noise_level))
      columns = ['mom','B','theta','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman69, Lecture II.15.4

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: -mom*B*cos(theta)
      """
      mom = X[0]
      B = X[1]
      theta = X[2]
      return Feynman69.calculate(mom,B,theta)

  @staticmethod
  def calculate(mom,B,theta):
      """
      Feynman69, Lecture II.15.4

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: -mom*B*cos(theta)
      """
      return -mom*B*np.cos(theta)
  

class Feynman70:
  equation_lambda = lambda args : (lambda p_d,Ef,theta: -p_d*Ef*np.cos(theta) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman70, Lecture II.15.5

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['p_d','Ef','theta','E_n']
      """
      p_d = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      return Feynman70.calculate_df(p_d,Ef,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(p_d,Ef,theta, noise_level = 0, include_original_target = False):
      """
      Feynman70, Lecture II.15.5

      Arguments:
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['p_d','Ef','theta','E_n']
      """
      target = Feynman70.calculate(p_d,Ef,theta)
      data = [p_d,Ef,theta]
      data.append(Noise(target,noise_level))
      columns = ['p_d','Ef','theta','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman70, Lecture II.15.5

      Arguments:
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: -p_d*Ef*cos(theta)
      """
      p_d = X[0]
      Ef = X[1]
      theta = X[2]
      return Feynman70.calculate(p_d,Ef,theta)

  @staticmethod
  def calculate(p_d,Ef,theta):
      """
      Feynman70, Lecture II.15.5

      Arguments:
          p_d: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: -p_d*Ef*cos(theta)
      """
      return -p_d*Ef*np.cos(theta)
  

class Feynman71:
  equation_lambda = lambda args : (lambda q,epsilon,r,v,c: q/(4*np.pi*epsilon*r*(1-v/c)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman71, Lecture II.21.32

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','r','v','c','Volt']
      """
      q = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(3.0,10.0, size)
      return Feynman71.calculate_df(q,epsilon,r,v,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,epsilon,r,v,c, noise_level = 0, include_original_target = False):
      """
      Feynman71, Lecture II.21.32

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','epsilon','r','v','c','Volt']
      """
      target = Feynman71.calculate(q,epsilon,r,v,c)
      data = [q,epsilon,r,v,c]
      data.append(Noise(target,noise_level))
      columns = ['q','epsilon','r','v','c','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman71, Lecture II.21.32

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: q/(4*pi*epsilon*r*(1-v/c))
      """
      q = X[0]
      epsilon = X[1]
      r = X[2]
      v = X[3]
      c = X[4]
      return Feynman71.calculate(q,epsilon,r,v,c)

  @staticmethod
  def calculate(q,epsilon,r,v,c):
      """
      Feynman71, Lecture II.21.32

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (3.0,10.0)
      Returns:
          f: q/(4*pi*epsilon*r*(1-v/c))
      """
      return q/(4*np.pi*epsilon*r*(1-v/c))
  

class Feynman72:
  equation_lambda = lambda args : (lambda omega,c,d: np.sqrt(omega**2/c**2-np.pi**2/d**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman72, Lecture II.24.17

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','c','d','k']
      """
      omega = np.random.uniform(4.0,6.0, size)
      c = np.random.uniform(1.0,2.0, size)
      d = np.random.uniform(2.0,4.0, size)
      return Feynman72.calculate_df(omega,c,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(omega,c,d, noise_level = 0, include_original_target = False):
      """
      Feynman72, Lecture II.24.17

      Arguments:
          omega: float or array-like, default range (4.0,6.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,4.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','c','d','k']
      """
      target = Feynman72.calculate(omega,c,d)
      data = [omega,c,d]
      data.append(Noise(target,noise_level))
      columns = ['omega','c','d','k']

      if(include_original_target):
         data.append(target)
         columns.append('k_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman72, Lecture II.24.17

      Arguments:
          omega: float or array-like, default range (4.0,6.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,4.0)
      Returns:
          f: sqrt(omega**2/c**2-pi**2/d**2)
      """
      omega = X[0]
      c = X[1]
      d = X[2]
      return Feynman72.calculate(omega,c,d)

  @staticmethod
  def calculate(omega,c,d):
      """
      Feynman72, Lecture II.24.17

      Arguments:
          omega: float or array-like, default range (4.0,6.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (2.0,4.0)
      Returns:
          f: sqrt(omega**2/c**2-pi**2/d**2)
      """
      return np.sqrt(omega**2/c**2-np.pi**2/d**2)
  

class Feynman73:
  equation_lambda = lambda args : (lambda epsilon,c,Ef: epsilon*c*Ef**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman73, Lecture II.27.16

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','Ef','flux']
      """
      epsilon = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      return Feynman73.calculate_df(epsilon,c,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,c,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman73, Lecture II.27.16

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','c','Ef','flux']
      """
      target = Feynman73.calculate(epsilon,c,Ef)
      data = [epsilon,c,Ef]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','c','Ef','flux']

      if(include_original_target):
         data.append(target)
         columns.append('flux_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman73, Lecture II.27.16

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*c*Ef**2
      """
      epsilon = X[0]
      c = X[1]
      Ef = X[2]
      return Feynman73.calculate(epsilon,c,Ef)

  @staticmethod
  def calculate(epsilon,c,Ef):
      """
      Feynman73, Lecture II.27.16

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*c*Ef**2
      """
      return epsilon*c*Ef**2
  

class Feynman74:
  equation_lambda = lambda args : (lambda epsilon,Ef: epsilon*Ef**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman74, Lecture II.27.18

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','Ef','E_den']
      """
      epsilon = np.random.uniform(1.0,5.0, size)
      Ef = np.random.uniform(1.0,5.0, size)
      return Feynman74.calculate_df(epsilon,Ef,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,Ef, noise_level = 0, include_original_target = False):
      """
      Feynman74, Lecture II.27.18

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','Ef','E_den']
      """
      target = Feynman74.calculate(epsilon,Ef)
      data = [epsilon,Ef]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','Ef','E_den']

      if(include_original_target):
         data.append(target)
         columns.append('E_den_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman74, Lecture II.27.18

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*Ef**2
      """
      epsilon = X[0]
      Ef = X[1]
      return Feynman74.calculate(epsilon,Ef)

  @staticmethod
  def calculate(epsilon,Ef):
      """
      Feynman74, Lecture II.27.18

      Arguments:
          epsilon: float or array-like, default range (1.0,5.0)
          Ef: float or array-like, default range (1.0,5.0)
      Returns:
          f: epsilon*Ef**2
      """
      return epsilon*Ef**2
  

class Feynman75:
  equation_lambda = lambda args : (lambda q,v,r: q*v/(2*np.pi*r) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman75, Lecture II.34.2a

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','r','I']
      """
      q = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman75.calculate_df(q,v,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,v,r, noise_level = 0, include_original_target = False):
      """
      Feynman75, Lecture II.34.2a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','r','I']
      """
      target = Feynman75.calculate(q,v,r)
      data = [q,v,r]
      data.append(Noise(target,noise_level))
      columns = ['q','v','r','I']

      if(include_original_target):
         data.append(target)
         columns.append('I_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman75, Lecture II.34.2a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v/(2*pi*r)
      """
      q = X[0]
      v = X[1]
      r = X[2]
      return Feynman75.calculate(q,v,r)

  @staticmethod
  def calculate(q,v,r):
      """
      Feynman75, Lecture II.34.2a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v/(2*pi*r)
      """
      return q*v/(2*np.pi*r)
  

class Feynman76:
  equation_lambda = lambda args : (lambda q,v,r: q*v*r/2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman76, Lecture II.34.2

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','r','mom']
      """
      q = np.random.uniform(1.0,5.0, size)
      v = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      return Feynman76.calculate_df(q,v,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,v,r, noise_level = 0, include_original_target = False):
      """
      Feynman76, Lecture II.34.2

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','v','r','mom']
      """
      target = Feynman76.calculate(q,v,r)
      data = [q,v,r]
      data.append(Noise(target,noise_level))
      columns = ['q','v','r','mom']

      if(include_original_target):
         data.append(target)
         columns.append('mom_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman76, Lecture II.34.2

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v*r/2
      """
      q = X[0]
      v = X[1]
      r = X[2]
      return Feynman76.calculate(q,v,r)

  @staticmethod
  def calculate(q,v,r):
      """
      Feynman76, Lecture II.34.2

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          v: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*v*r/2
      """
      return q*v*r/2
  

class Feynman77:
  equation_lambda = lambda args : (lambda g_,q,B,m: g_*q*B/(2*m) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman77, Lecture II.34.11

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['g_','q','B','m','omega']
      """
      g_ = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      m = np.random.uniform(1.0,5.0, size)
      return Feynman77.calculate_df(g_,q,B,m,noise_level,include_original_target)

  @staticmethod
  def calculate_df(g_,q,B,m, noise_level = 0, include_original_target = False):
      """
      Feynman77, Lecture II.34.11

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['g_','q','B','m','omega']
      """
      target = Feynman77.calculate(g_,q,B,m)
      data = [g_,q,B,m]
      data.append(Noise(target,noise_level))
      columns = ['g_','q','B','m','omega']

      if(include_original_target):
         data.append(target)
         columns.append('omega_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman77, Lecture II.34.11

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: g_*q*B/(2*m)
      """
      g_ = X[0]
      q = X[1]
      B = X[2]
      m = X[3]
      return Feynman77.calculate(g_,q,B,m)

  @staticmethod
  def calculate(g_,q,B,m):
      """
      Feynman77, Lecture II.34.11

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: g_*q*B/(2*m)
      """
      return g_*q*B/(2*m)
  

class Feynman78:
  equation_lambda = lambda args : (lambda q,h,m: q*h/(4*np.pi*m) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman78, Lecture II.34.29a

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','h','m','mom']
      """
      q = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      m = np.random.uniform(1.0,5.0, size)
      return Feynman78.calculate_df(q,h,m,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,h,m, noise_level = 0, include_original_target = False):
      """
      Feynman78, Lecture II.34.29a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','h','m','mom']
      """
      target = Feynman78.calculate(q,h,m)
      data = [q,h,m]
      data.append(Noise(target,noise_level))
      columns = ['q','h','m','mom']

      if(include_original_target):
         data.append(target)
         columns.append('mom_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman78, Lecture II.34.29a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*h/(4*pi*m)
      """
      q = X[0]
      h = X[1]
      m = X[2]
      return Feynman78.calculate(q,h,m)

  @staticmethod
  def calculate(q,h,m):
      """
      Feynman78, Lecture II.34.29a

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: q*h/(4*pi*m)
      """
      return q*h/(4*np.pi*m)
  

class Feynman79:
  equation_lambda = lambda args : (lambda g_,h,Jz,mom,B: g_*mom*B*Jz/(h/(2*np.pi)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman79, Lecture II.34.29b

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['g_','h','Jz','mom','B','E_n']
      """
      g_ = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      Jz = np.random.uniform(1.0,5.0, size)
      mom = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      return Feynman79.calculate_df(g_,h,Jz,mom,B,noise_level,include_original_target)

  @staticmethod
  def calculate_df(g_,h,Jz,mom,B, noise_level = 0, include_original_target = False):
      """
      Feynman79, Lecture II.34.29b

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          Jz: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['g_','h','Jz','mom','B','E_n']
      """
      target = Feynman79.calculate(g_,h,Jz,mom,B)
      data = [g_,h,Jz,mom,B]
      data.append(Noise(target,noise_level))
      columns = ['g_','h','Jz','mom','B','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman79, Lecture II.34.29b

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          Jz: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
      Returns:
          f: g_*mom*B*Jz/(h/(2*pi))
      """
      g_ = X[0]
      h = X[1]
      Jz = X[2]
      mom = X[3]
      B = X[4]
      return Feynman79.calculate(g_,h,Jz,mom,B)

  @staticmethod
  def calculate(g_,h,Jz,mom,B):
      """
      Feynman79, Lecture II.34.29b

      Arguments:
          g_: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          Jz: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
      Returns:
          f: g_*mom*B*Jz/(h/(2*pi))
      """
      return g_*mom*B*Jz/(h/(2*np.pi))
  

class Feynman80:
  equation_lambda = lambda args : (lambda n_0,kb,T,mom,B: n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T))) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman80, Lecture II.35.18

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','kb','T','mom','B','n']
      """
      n_0 = np.random.uniform(1.0,3.0, size)
      kb = np.random.uniform(1.0,3.0, size)
      T = np.random.uniform(1.0,3.0, size)
      mom = np.random.uniform(1.0,3.0, size)
      B = np.random.uniform(1.0,3.0, size)
      return Feynman80.calculate_df(n_0,kb,T,mom,B,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n_0,kb,T,mom,B, noise_level = 0, include_original_target = False):
      """
      Feynman80, Lecture II.35.18

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          mom: float or array-like, default range (1.0,3.0)
          B: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_0','kb','T','mom','B','n']
      """
      target = Feynman80.calculate(n_0,kb,T,mom,B)
      data = [n_0,kb,T,mom,B]
      data.append(Noise(target,noise_level))
      columns = ['n_0','kb','T','mom','B','n']

      if(include_original_target):
         data.append(target)
         columns.append('n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman80, Lecture II.35.18

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          mom: float or array-like, default range (1.0,3.0)
          B: float or array-like, default range (1.0,3.0)
      Returns:
          f: n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))
      """
      n_0 = X[0]
      kb = X[1]
      T = X[2]
      mom = X[3]
      B = X[4]
      return Feynman80.calculate(n_0,kb,T,mom,B)

  @staticmethod
  def calculate(n_0,kb,T,mom,B):
      """
      Feynman80, Lecture II.35.18

      Arguments:
          n_0: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          mom: float or array-like, default range (1.0,3.0)
          B: float or array-like, default range (1.0,3.0)
      Returns:
          f: n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))
      """
      return n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))
  

class Feynman81:
  equation_lambda = lambda args : (lambda n_rho,mom,B,kb,T: n_rho*mom*np.tanh(mom*B/(kb*T)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman81, Lecture II.35.21

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_rho','mom','B','kb','T','M']
      """
      n_rho = np.random.uniform(1.0,5.0, size)
      mom = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      return Feynman81.calculate_df(n_rho,mom,B,kb,T,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n_rho,mom,B,kb,T, noise_level = 0, include_original_target = False):
      """
      Feynman81, Lecture II.35.21

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n_rho','mom','B','kb','T','M']
      """
      target = Feynman81.calculate(n_rho,mom,B,kb,T)
      data = [n_rho,mom,B,kb,T]
      data.append(Noise(target,noise_level))
      columns = ['n_rho','mom','B','kb','T','M']

      if(include_original_target):
         data.append(target)
         columns.append('M_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman81, Lecture II.35.21

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_rho*mom*tanh(mom*B/(kb*T))
      """
      n_rho = X[0]
      mom = X[1]
      B = X[2]
      kb = X[3]
      T = X[4]
      return Feynman81.calculate(n_rho,mom,B,kb,T)

  @staticmethod
  def calculate(n_rho,mom,B,kb,T):
      """
      Feynman81, Lecture II.35.21

      Arguments:
          n_rho: float or array-like, default range (1.0,5.0)
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: n_rho*mom*tanh(mom*B/(kb*T))
      """
      return n_rho*mom*np.tanh(mom*B/(kb*T))
  

class Feynman82:
  equation_lambda = lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman82, Lecture II.36.38

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','H','kb','T','alpha','epsilon','c','M','f']
      """
      mom = np.random.uniform(1.0,3.0, size)
      H = np.random.uniform(1.0,3.0, size)
      kb = np.random.uniform(1.0,3.0, size)
      T = np.random.uniform(1.0,3.0, size)
      alpha = np.random.uniform(1.0,3.0, size)
      epsilon = np.random.uniform(1.0,3.0, size)
      c = np.random.uniform(1.0,3.0, size)
      M = np.random.uniform(1.0,3.0, size)
      return Feynman82.calculate_df(mom,H,kb,T,alpha,epsilon,c,M,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mom,H,kb,T,alpha,epsilon,c,M, noise_level = 0, include_original_target = False):
      """
      Feynman82, Lecture II.36.38

      Arguments:
          mom: float or array-like, default range (1.0,3.0)
          H: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          epsilon: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          M: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','H','kb','T','alpha','epsilon','c','M','f']
      """
      target = Feynman82.calculate(mom,H,kb,T,alpha,epsilon,c,M)
      data = [mom,H,kb,T,alpha,epsilon,c,M]
      data.append(Noise(target,noise_level))
      columns = ['mom','H','kb','T','alpha','epsilon','c','M','f']

      if(include_original_target):
         data.append(target)
         columns.append('f_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman82, Lecture II.36.38

      Arguments:
          mom: float or array-like, default range (1.0,3.0)
          H: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          epsilon: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          M: float or array-like, default range (1.0,3.0)
      Returns:
          f: mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
      """
      mom = X[0]
      H = X[1]
      kb = X[2]
      T = X[3]
      alpha = X[4]
      epsilon = X[5]
      c = X[6]
      M = X[7]
      return Feynman82.calculate(mom,H,kb,T,alpha,epsilon,c,M)

  @staticmethod
  def calculate(mom,H,kb,T,alpha,epsilon,c,M):
      """
      Feynman82, Lecture II.36.38

      Arguments:
          mom: float or array-like, default range (1.0,3.0)
          H: float or array-like, default range (1.0,3.0)
          kb: float or array-like, default range (1.0,3.0)
          T: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          epsilon: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          M: float or array-like, default range (1.0,3.0)
      Returns:
          f: mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
      """
      return mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
  

class Feynman83:
  equation_lambda = lambda args : (lambda mom,B,chi: mom*(1+chi)*B )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman83, Lecture II.37.1

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','chi','E_n']
      """
      mom = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      chi = np.random.uniform(1.0,5.0, size)
      return Feynman83.calculate_df(mom,B,chi,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mom,B,chi, noise_level = 0, include_original_target = False):
      """
      Feynman83, Lecture II.37.1

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','chi','E_n']
      """
      target = Feynman83.calculate(mom,B,chi)
      data = [mom,B,chi]
      data.append(Noise(target,noise_level))
      columns = ['mom','B','chi','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman83, Lecture II.37.1

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
      Returns:
          f: mom*(1+chi)*B
      """
      mom = X[0]
      B = X[1]
      chi = X[2]
      return Feynman83.calculate(mom,B,chi)

  @staticmethod
  def calculate(mom,B,chi):
      """
      Feynman83, Lecture II.37.1

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          chi: float or array-like, default range (1.0,5.0)
      Returns:
          f: mom*(1+chi)*B
      """
      return mom*(1+chi)*B
  

class Feynman84:
  equation_lambda = lambda args : (lambda Y,A,d,x: Y*A*x/d )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman84, Lecture II.38.3

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Y','A','d','x','F']
      """
      Y = np.random.uniform(1.0,5.0, size)
      A = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      x = np.random.uniform(1.0,5.0, size)
      return Feynman84.calculate_df(Y,A,d,x,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Y,A,d,x, noise_level = 0, include_original_target = False):
      """
      Feynman84, Lecture II.38.3

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Y','A','d','x','F']
      """
      target = Feynman84.calculate(Y,A,d,x)
      data = [Y,A,d,x]
      data.append(Noise(target,noise_level))
      columns = ['Y','A','d','x','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman84, Lecture II.38.3

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
      Returns:
          f: Y*A*x/d
      """
      Y = X[0]
      A = X[1]
      d = X[2]
      x = X[3]
      return Feynman84.calculate(Y,A,d,x)

  @staticmethod
  def calculate(Y,A,d,x):
      """
      Feynman84, Lecture II.38.3

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          A: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
      Returns:
          f: Y*A*x/d
      """
      return Y*A*x/d
  

class Feynman85:
  equation_lambda = lambda args : (lambda Y,sigma: Y/(2*(1+sigma)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman85, Lecture II.38.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Y','sigma','mu_S']
      """
      Y = np.random.uniform(1.0,5.0, size)
      sigma = np.random.uniform(1.0,5.0, size)
      return Feynman85.calculate_df(Y,sigma,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Y,sigma, noise_level = 0, include_original_target = False):
      """
      Feynman85, Lecture II.38.14

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          sigma: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Y','sigma','mu_S']
      """
      target = Feynman85.calculate(Y,sigma)
      data = [Y,sigma]
      data.append(Noise(target,noise_level))
      columns = ['Y','sigma','mu_S']

      if(include_original_target):
         data.append(target)
         columns.append('mu_S_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman85, Lecture II.38.14

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          sigma: float or array-like, default range (1.0,5.0)
      Returns:
          f: Y/(2*(1+sigma))
      """
      Y = X[0]
      sigma = X[1]
      return Feynman85.calculate(Y,sigma)

  @staticmethod
  def calculate(Y,sigma):
      """
      Feynman85, Lecture II.38.14

      Arguments:
          Y: float or array-like, default range (1.0,5.0)
          sigma: float or array-like, default range (1.0,5.0)
      Returns:
          f: Y/(2*(1+sigma))
      """
      return Y/(2*(1+sigma))
  

class Feynman86:
  equation_lambda = lambda args : (lambda h,omega,kb,T: 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman86, Lecture III.4.32

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','omega','kb','T','n']
      """
      h = np.random.uniform(1.0,5.0, size)
      omega = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      return Feynman86.calculate_df(h,omega,kb,T,noise_level,include_original_target)

  @staticmethod
  def calculate_df(h,omega,kb,T, noise_level = 0, include_original_target = False):
      """
      Feynman86, Lecture III.4.32

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','omega','kb','T','n']
      """
      target = Feynman86.calculate(h,omega,kb,T)
      data = [h,omega,kb,T]
      data.append(Noise(target,noise_level))
      columns = ['h','omega','kb','T','n']

      if(include_original_target):
         data.append(target)
         columns.append('n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman86, Lecture III.4.32

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(exp((h/(2*pi))*omega/(kb*T))-1)
      """
      h = X[0]
      omega = X[1]
      kb = X[2]
      T = X[3]
      return Feynman86.calculate(h,omega,kb,T)

  @staticmethod
  def calculate(h,omega,kb,T):
      """
      Feynman86, Lecture III.4.32

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(exp((h/(2*pi))*omega/(kb*T))-1)
      """
      return 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
  

class Feynman87:
  equation_lambda = lambda args : (lambda h,omega,kb,T: (h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman87, Lecture III.4.33

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','omega','kb','T','E_n']
      """
      h = np.random.uniform(1.0,5.0, size)
      omega = np.random.uniform(1.0,5.0, size)
      kb = np.random.uniform(1.0,5.0, size)
      T = np.random.uniform(1.0,5.0, size)
      return Feynman87.calculate_df(h,omega,kb,T,noise_level,include_original_target)

  @staticmethod
  def calculate_df(h,omega,kb,T, noise_level = 0, include_original_target = False):
      """
      Feynman87, Lecture III.4.33

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','omega','kb','T','E_n']
      """
      target = Feynman87.calculate(h,omega,kb,T)
      data = [h,omega,kb,T]
      data.append(Noise(target,noise_level))
      columns = ['h','omega','kb','T','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman87, Lecture III.4.33

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)
      """
      h = X[0]
      omega = X[1]
      kb = X[2]
      T = X[3]
      return Feynman87.calculate(h,omega,kb,T)

  @staticmethod
  def calculate(h,omega,kb,T):
      """
      Feynman87, Lecture III.4.33

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          kb: float or array-like, default range (1.0,5.0)
          T: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)
      """
      return (h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
  

class Feynman88:
  equation_lambda = lambda args : (lambda mom,B,h: 2*mom*B/(h/(2*np.pi)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman88, Lecture III.7.38

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','h','omega']
      """
      mom = np.random.uniform(1.0,5.0, size)
      B = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      return Feynman88.calculate_df(mom,B,h,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mom,B,h, noise_level = 0, include_original_target = False):
      """
      Feynman88, Lecture III.7.38

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','B','h','omega']
      """
      target = Feynman88.calculate(mom,B,h)
      data = [mom,B,h]
      data.append(Noise(target,noise_level))
      columns = ['mom','B','h','omega']

      if(include_original_target):
         data.append(target)
         columns.append('omega_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman88, Lecture III.7.38

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*mom*B/(h/(2*pi))
      """
      mom = X[0]
      B = X[1]
      h = X[2]
      return Feynman88.calculate(mom,B,h)

  @staticmethod
  def calculate(mom,B,h):
      """
      Feynman88, Lecture III.7.38

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          B: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*mom*B/(h/(2*pi))
      """
      return 2*mom*B/(h/(2*np.pi))
  

class Feynman89:
  equation_lambda = lambda args : (lambda E_n,t,h: np.sin(E_n*t/(h/(2*np.pi)))**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman89, Lecture III.8.54

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','t','h','prob']
      """
      E_n = np.random.uniform(1.0,2.0, size)
      t = np.random.uniform(1.0,2.0, size)
      h = np.random.uniform(1.0,4.0, size)
      return Feynman89.calculate_df(E_n,t,h,noise_level,include_original_target)

  @staticmethod
  def calculate_df(E_n,t,h, noise_level = 0, include_original_target = False):
      """
      Feynman89, Lecture III.8.54

      Arguments:
          E_n: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,2.0)
          h: float or array-like, default range (1.0,4.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','t','h','prob']
      """
      target = Feynman89.calculate(E_n,t,h)
      data = [E_n,t,h]
      data.append(Noise(target,noise_level))
      columns = ['E_n','t','h','prob']

      if(include_original_target):
         data.append(target)
         columns.append('prob_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman89, Lecture III.8.54

      Arguments:
          E_n: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,2.0)
          h: float or array-like, default range (1.0,4.0)
      Returns:
          f: sin(E_n*t/(h/(2*pi)))**2
      """
      E_n = X[0]
      t = X[1]
      h = X[2]
      return Feynman89.calculate(E_n,t,h)

  @staticmethod
  def calculate(E_n,t,h):
      """
      Feynman89, Lecture III.8.54

      Arguments:
          E_n: float or array-like, default range (1.0,2.0)
          t: float or array-like, default range (1.0,2.0)
          h: float or array-like, default range (1.0,4.0)
      Returns:
          f: sin(E_n*t/(h/(2*pi)))**2
      """
      return np.sin(E_n*t/(h/(2*np.pi)))**2
  

class Feynman90:
  equation_lambda = lambda args : (lambda p_d,Ef,t,h,omega,omega_0: (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman90, Lecture III.9.52

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['p_d','Ef','t','h','omega','omega_0','prob']
      """
      p_d = np.random.uniform(1.0,3.0, size)
      Ef = np.random.uniform(1.0,3.0, size)
      t = np.random.uniform(1.0,3.0, size)
      h = np.random.uniform(1.0,3.0, size)
      omega = np.random.uniform(1.0,5.0, size)
      omega_0 = np.random.uniform(1.0,5.0, size)
      return Feynman90.calculate_df(p_d,Ef,t,h,omega,omega_0,noise_level,include_original_target)

  @staticmethod
  def calculate_df(p_d,Ef,t,h,omega,omega_0, noise_level = 0, include_original_target = False):
      """
      Feynman90, Lecture III.9.52

      Arguments:
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          h: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['p_d','Ef','t','h','omega','omega_0','prob']
      """
      target = Feynman90.calculate(p_d,Ef,t,h,omega,omega_0)
      data = [p_d,Ef,t,h,omega,omega_0]
      data.append(Noise(target,noise_level))
      columns = ['p_d','Ef','t','h','omega','omega_0','prob']

      if(include_original_target):
         data.append(target)
         columns.append('prob_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman90, Lecture III.9.52

      Arguments:
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          h: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
      """
      p_d = X[0]
      Ef = X[1]
      t = X[2]
      h = X[3]
      omega = X[4]
      omega_0 = X[5]
      return Feynman90.calculate(p_d,Ef,t,h,omega,omega_0)

  @staticmethod
  def calculate(p_d,Ef,t,h,omega,omega_0):
      """
      Feynman90, Lecture III.9.52

      Arguments:
          p_d: float or array-like, default range (1.0,3.0)
          Ef: float or array-like, default range (1.0,3.0)
          t: float or array-like, default range (1.0,3.0)
          h: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
      Returns:
          f: (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
      """
      return (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
  

class Feynman91:
  equation_lambda = lambda args : (lambda mom,Bx,By,Bz: mom*np.sqrt(Bx**2+By**2+Bz**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman91, Lecture III.10.19

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','Bx','By','Bz','E_n']
      """
      mom = np.random.uniform(1.0,5.0, size)
      Bx = np.random.uniform(1.0,5.0, size)
      By = np.random.uniform(1.0,5.0, size)
      Bz = np.random.uniform(1.0,5.0, size)
      return Feynman91.calculate_df(mom,Bx,By,Bz,noise_level,include_original_target)

  @staticmethod
  def calculate_df(mom,Bx,By,Bz, noise_level = 0, include_original_target = False):
      """
      Feynman91, Lecture III.10.19

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          Bx: float or array-like, default range (1.0,5.0)
          By: float or array-like, default range (1.0,5.0)
          Bz: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['mom','Bx','By','Bz','E_n']
      """
      target = Feynman91.calculate(mom,Bx,By,Bz)
      data = [mom,Bx,By,Bz]
      data.append(Noise(target,noise_level))
      columns = ['mom','Bx','By','Bz','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman91, Lecture III.10.19

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          Bx: float or array-like, default range (1.0,5.0)
          By: float or array-like, default range (1.0,5.0)
          Bz: float or array-like, default range (1.0,5.0)
      Returns:
          f: mom*sqrt(Bx**2+By**2+Bz**2)
      """
      mom = X[0]
      Bx = X[1]
      By = X[2]
      Bz = X[3]
      return Feynman91.calculate(mom,Bx,By,Bz)

  @staticmethod
  def calculate(mom,Bx,By,Bz):
      """
      Feynman91, Lecture III.10.19

      Arguments:
          mom: float or array-like, default range (1.0,5.0)
          Bx: float or array-like, default range (1.0,5.0)
          By: float or array-like, default range (1.0,5.0)
          Bz: float or array-like, default range (1.0,5.0)
      Returns:
          f: mom*sqrt(Bx**2+By**2+Bz**2)
      """
      return mom*np.sqrt(Bx**2+By**2+Bz**2)
  

class Feynman92:
  equation_lambda = lambda args : (lambda n,h: n*(h/(2*np.pi)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman92, Lecture III.12.43

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','h','L']
      """
      n = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      return Feynman92.calculate_df(n,h,noise_level,include_original_target)

  @staticmethod
  def calculate_df(n,h, noise_level = 0, include_original_target = False):
      """
      Feynman92, Lecture III.12.43

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['n','h','L']
      """
      target = Feynman92.calculate(n,h)
      data = [n,h]
      data.append(Noise(target,noise_level))
      columns = ['n','h','L']

      if(include_original_target):
         data.append(target)
         columns.append('L_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman92, Lecture III.12.43

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*(h/(2*pi))
      """
      n = X[0]
      h = X[1]
      return Feynman92.calculate(n,h)

  @staticmethod
  def calculate(n,h):
      """
      Feynman92, Lecture III.12.43

      Arguments:
          n: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: n*(h/(2*pi))
      """
      return n*(h/(2*np.pi))
  

class Feynman93:
  equation_lambda = lambda args : (lambda E_n,d,k,h: 2*E_n*d**2*k/(h/(2*np.pi)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman93, Lecture III.13.18

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','d','k','h','v']
      """
      E_n = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      k = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      return Feynman93.calculate_df(E_n,d,k,h,noise_level,include_original_target)

  @staticmethod
  def calculate_df(E_n,d,k,h, noise_level = 0, include_original_target = False):
      """
      Feynman93, Lecture III.13.18

      Arguments:
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','d','k','h','v']
      """
      target = Feynman93.calculate(E_n,d,k,h)
      data = [E_n,d,k,h]
      data.append(Noise(target,noise_level))
      columns = ['E_n','d','k','h','v']

      if(include_original_target):
         data.append(target)
         columns.append('v_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman93, Lecture III.13.18

      Arguments:
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*E_n*d**2*k/(h/(2*pi))
      """
      E_n = X[0]
      d = X[1]
      k = X[2]
      h = X[3]
      return Feynman93.calculate(E_n,d,k,h)

  @staticmethod
  def calculate(E_n,d,k,h):
      """
      Feynman93, Lecture III.13.18

      Arguments:
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*E_n*d**2*k/(h/(2*pi))
      """
      return 2*E_n*d**2*k/(h/(2*np.pi))
  

class Feynman94:
  equation_lambda = lambda args : (lambda I_0,q,Volt,kb,T: I_0*(np.exp(q*Volt/(kb*T))-1) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman94, Lecture III.14.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I_0','q','Volt','kb','T','I']
      """
      I_0 = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,2.0, size)
      Volt = np.random.uniform(1.0,2.0, size)
      kb = np.random.uniform(1.0,2.0, size)
      T = np.random.uniform(1.0,2.0, size)
      return Feynman94.calculate_df(I_0,q,Volt,kb,T,noise_level,include_original_target)

  @staticmethod
  def calculate_df(I_0,q,Volt,kb,T, noise_level = 0, include_original_target = False):
      """
      Feynman94, Lecture III.14.14

      Arguments:
          I_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,2.0)
          Volt: float or array-like, default range (1.0,2.0)
          kb: float or array-like, default range (1.0,2.0)
          T: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I_0','q','Volt','kb','T','I']
      """
      target = Feynman94.calculate(I_0,q,Volt,kb,T)
      data = [I_0,q,Volt,kb,T]
      data.append(Noise(target,noise_level))
      columns = ['I_0','q','Volt','kb','T','I']

      if(include_original_target):
         data.append(target)
         columns.append('I_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman94, Lecture III.14.14

      Arguments:
          I_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,2.0)
          Volt: float or array-like, default range (1.0,2.0)
          kb: float or array-like, default range (1.0,2.0)
          T: float or array-like, default range (1.0,2.0)
      Returns:
          f: I_0*(exp(q*Volt/(kb*T))-1)
      """
      I_0 = X[0]
      q = X[1]
      Volt = X[2]
      kb = X[3]
      T = X[4]
      return Feynman94.calculate(I_0,q,Volt,kb,T)

  @staticmethod
  def calculate(I_0,q,Volt,kb,T):
      """
      Feynman94, Lecture III.14.14

      Arguments:
          I_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,2.0)
          Volt: float or array-like, default range (1.0,2.0)
          kb: float or array-like, default range (1.0,2.0)
          T: float or array-like, default range (1.0,2.0)
      Returns:
          f: I_0*(exp(q*Volt/(kb*T))-1)
      """
      return I_0*(np.exp(q*Volt/(kb*T))-1)
  

class Feynman95:
  equation_lambda = lambda args : (lambda U,k,d: 2*U*(1-np.cos(k*d)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman95, Lecture III.15.12

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['U','k','d','E_n']
      """
      U = np.random.uniform(1.0,5.0, size)
      k = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman95.calculate_df(U,k,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(U,k,d, noise_level = 0, include_original_target = False):
      """
      Feynman95, Lecture III.15.12

      Arguments:
          U: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['U','k','d','E_n']
      """
      target = Feynman95.calculate(U,k,d)
      data = [U,k,d]
      data.append(Noise(target,noise_level))
      columns = ['U','k','d','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman95, Lecture III.15.12

      Arguments:
          U: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*U*(1-cos(k*d))
      """
      U = X[0]
      k = X[1]
      d = X[2]
      return Feynman95.calculate(U,k,d)

  @staticmethod
  def calculate(U,k,d):
      """
      Feynman95, Lecture III.15.12

      Arguments:
          U: float or array-like, default range (1.0,5.0)
          k: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*U*(1-cos(k*d))
      """
      return 2*U*(1-np.cos(k*d))
  

class Feynman96:
  equation_lambda = lambda args : (lambda h,E_n,d: (h/(2*np.pi))**2/(2*E_n*d**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman96, Lecture III.15.14

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','E_n','d','m']
      """
      h = np.random.uniform(1.0,5.0, size)
      E_n = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman96.calculate_df(h,E_n,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(h,E_n,d, noise_level = 0, include_original_target = False):
      """
      Feynman96, Lecture III.15.14

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['h','E_n','d','m']
      """
      target = Feynman96.calculate(h,E_n,d)
      data = [h,E_n,d]
      data.append(Noise(target,noise_level))
      columns = ['h','E_n','d','m']

      if(include_original_target):
         data.append(target)
         columns.append('m_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman96, Lecture III.15.14

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))**2/(2*E_n*d**2)
      """
      h = X[0]
      E_n = X[1]
      d = X[2]
      return Feynman96.calculate(h,E_n,d)

  @staticmethod
  def calculate(h,E_n,d):
      """
      Feynman96, Lecture III.15.14

      Arguments:
          h: float or array-like, default range (1.0,5.0)
          E_n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: (h/(2*pi))**2/(2*E_n*d**2)
      """
      return (h/(2*np.pi))**2/(2*E_n*d**2)
  

class Feynman97:
  equation_lambda = lambda args : (lambda alpha,n,d: 2*np.pi*alpha/(n*d) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman97, Lecture III.15.27

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['alpha','n','d','k']
      """
      alpha = np.random.uniform(1.0,5.0, size)
      n = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      return Feynman97.calculate_df(alpha,n,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(alpha,n,d, noise_level = 0, include_original_target = False):
      """
      Feynman97, Lecture III.15.27

      Arguments:
          alpha: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['alpha','n','d','k']
      """
      target = Feynman97.calculate(alpha,n,d)
      data = [alpha,n,d]
      data.append(Noise(target,noise_level))
      columns = ['alpha','n','d','k']

      if(include_original_target):
         data.append(target)
         columns.append('k_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman97, Lecture III.15.27

      Arguments:
          alpha: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*pi*alpha/(n*d)
      """
      alpha = X[0]
      n = X[1]
      d = X[2]
      return Feynman97.calculate(alpha,n,d)

  @staticmethod
  def calculate(alpha,n,d):
      """
      Feynman97, Lecture III.15.27

      Arguments:
          alpha: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
      Returns:
          f: 2*pi*alpha/(n*d)
      """
      return 2*np.pi*alpha/(n*d)
  

class Feynman98:
  equation_lambda = lambda args : (lambda beta,alpha,theta: beta*(1+alpha*np.cos(theta)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman98, Lecture III.17.37

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['beta','alpha','theta','f']
      """
      beta = np.random.uniform(1.0,5.0, size)
      alpha = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(1.0,5.0, size)
      return Feynman98.calculate_df(beta,alpha,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(beta,alpha,theta, noise_level = 0, include_original_target = False):
      """
      Feynman98, Lecture III.17.37

      Arguments:
          beta: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['beta','alpha','theta','f']
      """
      target = Feynman98.calculate(beta,alpha,theta)
      data = [beta,alpha,theta]
      data.append(Noise(target,noise_level))
      columns = ['beta','alpha','theta','f']

      if(include_original_target):
         data.append(target)
         columns.append('f_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman98, Lecture III.17.37

      Arguments:
          beta: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: beta*(1+alpha*cos(theta))
      """
      beta = X[0]
      alpha = X[1]
      theta = X[2]
      return Feynman98.calculate(beta,alpha,theta)

  @staticmethod
  def calculate(beta,alpha,theta):
      """
      Feynman98, Lecture III.17.37

      Arguments:
          beta: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (1.0,5.0)
      Returns:
          f: beta*(1+alpha*cos(theta))
      """
      return beta*(1+alpha*np.cos(theta))
  

class Feynman99:
  equation_lambda = lambda args : (lambda m,q,h,n,epsilon: -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman99, Lecture III.19.51

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','q','h','n','epsilon','E_n']
      """
      m = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      n = np.random.uniform(1.0,5.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      return Feynman99.calculate_df(m,q,h,n,epsilon,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,q,h,n,epsilon, noise_level = 0, include_original_target = False):
      """
      Feynman99, Lecture III.19.51

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','q','h','n','epsilon','E_n']
      """
      target = Feynman99.calculate(m,q,h,n,epsilon)
      data = [m,q,h,n,epsilon]
      data.append(Noise(target,noise_level))
      columns = ['m','q','h','n','epsilon','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman99, Lecture III.19.51

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)
      """
      m = X[0]
      q = X[1]
      h = X[2]
      n = X[3]
      epsilon = X[4]
      return Feynman99.calculate(m,q,h,n,epsilon)

  @staticmethod
  def calculate(m,q,h,n,epsilon):
      """
      Feynman99, Lecture III.19.51

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          n: float or array-like, default range (1.0,5.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)
      """
      return -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)
  

class Feynman100:
  equation_lambda = lambda args : (lambda rho_c_0,q,A_vec,m: -rho_c_0*q*A_vec/m )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Feynman100, Lecture III.21.20

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','q','A_vec','m','j']
      """
      rho_c_0 = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      A_vec = np.random.uniform(1.0,5.0, size)
      m = np.random.uniform(1.0,5.0, size)
      return Feynman100.calculate_df(rho_c_0,q,A_vec,m,noise_level,include_original_target)

  @staticmethod
  def calculate_df(rho_c_0,q,A_vec,m, noise_level = 0, include_original_target = False):
      """
      Feynman100, Lecture III.21.20

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['rho_c_0','q','A_vec','m','j']
      """
      target = Feynman100.calculate(rho_c_0,q,A_vec,m)
      data = [rho_c_0,q,A_vec,m]
      data.append(Noise(target,noise_level))
      columns = ['rho_c_0','q','A_vec','m','j']

      if(include_original_target):
         data.append(target)
         columns.append('j_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Feynman100, Lecture III.21.20

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: -rho_c_0*q*A_vec/m
      """
      rho_c_0 = X[0]
      q = X[1]
      A_vec = X[2]
      m = X[3]
      return Feynman100.calculate(rho_c_0,q,A_vec,m)

  @staticmethod
  def calculate(rho_c_0,q,A_vec,m):
      """
      Feynman100, Lecture III.21.20

      Arguments:
          rho_c_0: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
      Returns:
          f: -rho_c_0*q*A_vec/m
      """
      return -rho_c_0*q*A_vec/m
  

class Bonus1:
  equation_lambda = lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: (Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus1.0, Rutherford scattering

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Z_1','Z_2','alpha','hbar','c','E_n','theta','A']
      """
      Z_1 = np.random.uniform(1.0,2.0, size)
      Z_2 = np.random.uniform(1.0,2.0, size)
      alpha = np.random.uniform(1.0,2.0, size)
      hbar = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(1.0,2.0, size)
      E_n = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      return Bonus1.calculate_df(Z_1,Z_2,alpha,hbar,c,E_n,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Z_1,Z_2,alpha,hbar,c,E_n,theta, noise_level = 0, include_original_target = False):
      """
      Bonus1.0, Rutherford scattering

      Arguments:
          Z_1: float or array-like, default range (1.0,2.0)
          Z_2: float or array-like, default range (1.0,2.0)
          alpha: float or array-like, default range (1.0,2.0)
          hbar: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Z_1','Z_2','alpha','hbar','c','E_n','theta','A']
      """
      target = Bonus1.calculate(Z_1,Z_2,alpha,hbar,c,E_n,theta)
      data = [Z_1,Z_2,alpha,hbar,c,E_n,theta]
      data.append(Noise(target,noise_level))
      columns = ['Z_1','Z_2','alpha','hbar','c','E_n','theta','A']

      if(include_original_target):
         data.append(target)
         columns.append('A_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus1.0, Rutherford scattering

      Arguments:
          Z_1: float or array-like, default range (1.0,2.0)
          Z_2: float or array-like, default range (1.0,2.0)
          alpha: float or array-like, default range (1.0,2.0)
          hbar: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: (Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2
      """
      Z_1 = X[0]
      Z_2 = X[1]
      alpha = X[2]
      hbar = X[3]
      c = X[4]
      E_n = X[5]
      theta = X[6]
      return Bonus1.calculate(Z_1,Z_2,alpha,hbar,c,E_n,theta)

  @staticmethod
  def calculate(Z_1,Z_2,alpha,hbar,c,E_n,theta):
      """
      Bonus1.0, Rutherford scattering

      Arguments:
          Z_1: float or array-like, default range (1.0,2.0)
          Z_2: float or array-like, default range (1.0,2.0)
          alpha: float or array-like, default range (1.0,2.0)
          hbar: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: (Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2
      """
      return (Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2
  

class Bonus2:
  equation_lambda = lambda args : (lambda m,k_G,L,E_n,theta1,theta2: m*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus2.0, 3.55 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','k_G','L','E_n','theta1','theta2','k']
      """
      m = np.random.uniform(1.0,3.0, size)
      k_G = np.random.uniform(1.0,3.0, size)
      L = np.random.uniform(1.0,3.0, size)
      E_n = np.random.uniform(1.0,3.0, size)
      theta1 = np.random.uniform(0.0,6.0, size)
      theta2 = np.random.uniform(0.0,6.0, size)
      return Bonus2.calculate_df(m,k_G,L,E_n,theta1,theta2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,k_G,L,E_n,theta1,theta2, noise_level = 0, include_original_target = False):
      """
      Bonus2.0, 3.55 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          k_G: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (0.0,6.0)
          theta2: float or array-like, default range (0.0,6.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','k_G','L','E_n','theta1','theta2','k']
      """
      target = Bonus2.calculate(m,k_G,L,E_n,theta1,theta2)
      data = [m,k_G,L,E_n,theta1,theta2]
      data.append(Noise(target,noise_level))
      columns = ['m','k_G','L','E_n','theta1','theta2','k']

      if(include_original_target):
         data.append(target)
         columns.append('k_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus2.0, 3.55 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          k_G: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (0.0,6.0)
          theta2: float or array-like, default range (0.0,6.0)
      Returns:
          f: m*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))
      """
      m = X[0]
      k_G = X[1]
      L = X[2]
      E_n = X[3]
      theta1 = X[4]
      theta2 = X[5]
      return Bonus2.calculate(m,k_G,L,E_n,theta1,theta2)

  @staticmethod
  def calculate(m,k_G,L,E_n,theta1,theta2):
      """
      Bonus2.0, 3.55 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          k_G: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
          theta1: float or array-like, default range (0.0,6.0)
          theta2: float or array-like, default range (0.0,6.0)
      Returns:
          f: m*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))
      """
      return m*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2))
  

class Bonus3:
  equation_lambda = lambda args : (lambda d,alpha,theta1,theta2: d*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus3.0, 3.64 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d','alpha','theta1','theta2','r']
      """
      d = np.random.uniform(1.0,3.0, size)
      alpha = np.random.uniform(2.0,4.0, size)
      theta1 = np.random.uniform(4.0,5.0, size)
      theta2 = np.random.uniform(4.0,5.0, size)
      return Bonus3.calculate_df(d,alpha,theta1,theta2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(d,alpha,theta1,theta2, noise_level = 0, include_original_target = False):
      """
      Bonus3.0, 3.64 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (2.0,4.0)
          theta1: float or array-like, default range (4.0,5.0)
          theta2: float or array-like, default range (4.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d','alpha','theta1','theta2','r']
      """
      target = Bonus3.calculate(d,alpha,theta1,theta2)
      data = [d,alpha,theta1,theta2]
      data.append(Noise(target,noise_level))
      columns = ['d','alpha','theta1','theta2','r']

      if(include_original_target):
         data.append(target)
         columns.append('r_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus3.0, 3.64 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (2.0,4.0)
          theta1: float or array-like, default range (4.0,5.0)
          theta2: float or array-like, default range (4.0,5.0)
      Returns:
          f: d*(1-alpha**2)/(1+alpha*cos(theta1-theta2))
      """
      d = X[0]
      alpha = X[1]
      theta1 = X[2]
      theta2 = X[3]
      return Bonus3.calculate(d,alpha,theta1,theta2)

  @staticmethod
  def calculate(d,alpha,theta1,theta2):
      """
      Bonus3.0, 3.64 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (2.0,4.0)
          theta1: float or array-like, default range (4.0,5.0)
          theta2: float or array-like, default range (4.0,5.0)
      Returns:
          f: d*(1-alpha**2)/(1+alpha*cos(theta1-theta2))
      """
      return d*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2))
  

class Bonus4:
  equation_lambda = lambda args : (lambda m,E_n,U,L,r: np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2))) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus4.0, 3.16 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','E_n','U','L','r','v']
      """
      m = np.random.uniform(1.0,3.0, size)
      E_n = np.random.uniform(8.0,12.0, size)
      U = np.random.uniform(1.0,3.0, size)
      L = np.random.uniform(1.0,3.0, size)
      r = np.random.uniform(1.0,3.0, size)
      return Bonus4.calculate_df(m,E_n,U,L,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,E_n,U,L,r, noise_level = 0, include_original_target = False):
      """
      Bonus4.0, 3.16 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (8.0,12.0)
          U: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','E_n','U','L','r','v']
      """
      target = Bonus4.calculate(m,E_n,U,L,r)
      data = [m,E_n,U,L,r]
      data.append(Noise(target,noise_level))
      columns = ['m','E_n','U','L','r','v']

      if(include_original_target):
         data.append(target)
         columns.append('v_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus4.0, 3.16 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (8.0,12.0)
          U: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))
      """
      m = X[0]
      E_n = X[1]
      U = X[2]
      L = X[3]
      r = X[4]
      return Bonus4.calculate(m,E_n,U,L,r)

  @staticmethod
  def calculate(m,E_n,U,L,r):
      """
      Bonus4.0, 3.16 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (8.0,12.0)
          U: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          r: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))
      """
      return np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))
  

class Bonus5:
  equation_lambda = lambda args : (lambda d,G,m1,m2: 2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus5.0, 3.74 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d','G','m1','m2','t']
      """
      d = np.random.uniform(1.0,3.0, size)
      G = np.random.uniform(1.0,3.0, size)
      m1 = np.random.uniform(1.0,3.0, size)
      m2 = np.random.uniform(1.0,3.0, size)
      return Bonus5.calculate_df(d,G,m1,m2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(d,G,m1,m2, noise_level = 0, include_original_target = False):
      """
      Bonus5.0, 3.74 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          G: float or array-like, default range (1.0,3.0)
          m1: float or array-like, default range (1.0,3.0)
          m2: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['d','G','m1','m2','t']
      """
      target = Bonus5.calculate(d,G,m1,m2)
      data = [d,G,m1,m2]
      data.append(Noise(target,noise_level))
      columns = ['d','G','m1','m2','t']

      if(include_original_target):
         data.append(target)
         columns.append('t_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus5.0, 3.74 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          G: float or array-like, default range (1.0,3.0)
          m1: float or array-like, default range (1.0,3.0)
          m2: float or array-like, default range (1.0,3.0)
      Returns:
          f: 2*pi*d**(3/2)/sqrt(G*(m1+m2))
      """
      d = X[0]
      G = X[1]
      m1 = X[2]
      m2 = X[3]
      return Bonus5.calculate(d,G,m1,m2)

  @staticmethod
  def calculate(d,G,m1,m2):
      """
      Bonus5.0, 3.74 Goldstein

      Arguments:
          d: float or array-like, default range (1.0,3.0)
          G: float or array-like, default range (1.0,3.0)
          m1: float or array-like, default range (1.0,3.0)
          m2: float or array-like, default range (1.0,3.0)
      Returns:
          f: 2*pi*d**(3/2)/sqrt(G*(m1+m2))
      """
      return 2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2))
  

class Bonus6:
  equation_lambda = lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus6.0, 3.99 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','L','m','Z_1','Z_2','q','E_n','alpha']
      """
      epsilon = np.random.uniform(1.0,3.0, size)
      L = np.random.uniform(1.0,3.0, size)
      m = np.random.uniform(1.0,3.0, size)
      Z_1 = np.random.uniform(1.0,3.0, size)
      Z_2 = np.random.uniform(1.0,3.0, size)
      q = np.random.uniform(1.0,3.0, size)
      E_n = np.random.uniform(1.0,3.0, size)
      return Bonus6.calculate_df(epsilon,L,m,Z_1,Z_2,q,E_n,noise_level,include_original_target)

  @staticmethod
  def calculate_df(epsilon,L,m,Z_1,Z_2,q,E_n, noise_level = 0, include_original_target = False):
      """
      Bonus6.0, 3.99 Goldstein

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          Z_1: float or array-like, default range (1.0,3.0)
          Z_2: float or array-like, default range (1.0,3.0)
          q: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['epsilon','L','m','Z_1','Z_2','q','E_n','alpha']
      """
      target = Bonus6.calculate(epsilon,L,m,Z_1,Z_2,q,E_n)
      data = [epsilon,L,m,Z_1,Z_2,q,E_n]
      data.append(Noise(target,noise_level))
      columns = ['epsilon','L','m','Z_1','Z_2','q','E_n','alpha']

      if(include_original_target):
         data.append(target)
         columns.append('alpha_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus6.0, 3.99 Goldstein

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          Z_1: float or array-like, default range (1.0,3.0)
          Z_2: float or array-like, default range (1.0,3.0)
          q: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))
      """
      epsilon = X[0]
      L = X[1]
      m = X[2]
      Z_1 = X[3]
      Z_2 = X[4]
      q = X[5]
      E_n = X[6]
      return Bonus6.calculate(epsilon,L,m,Z_1,Z_2,q,E_n)

  @staticmethod
  def calculate(epsilon,L,m,Z_1,Z_2,q,E_n):
      """
      Bonus6.0, 3.99 Goldstein

      Arguments:
          epsilon: float or array-like, default range (1.0,3.0)
          L: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          Z_1: float or array-like, default range (1.0,3.0)
          Z_2: float or array-like, default range (1.0,3.0)
          q: float or array-like, default range (1.0,3.0)
          E_n: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))
      """
      return np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))
  

class Bonus7:
  equation_lambda = lambda args : (lambda G,rho,alpha,c,d: np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus7.0, Friedman Equation

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','rho','alpha','c','d','H_G']
      """
      G = np.random.uniform(1.0,3.0, size)
      rho = np.random.uniform(1.0,3.0, size)
      alpha = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(1.0,2.0, size)
      d = np.random.uniform(1.0,3.0, size)
      return Bonus7.calculate_df(G,rho,alpha,c,d,noise_level,include_original_target)

  @staticmethod
  def calculate_df(G,rho,alpha,c,d, noise_level = 0, include_original_target = False):
      """
      Bonus7.0, Friedman Equation

      Arguments:
          G: float or array-like, default range (1.0,3.0)
          rho: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','rho','alpha','c','d','H_G']
      """
      target = Bonus7.calculate(G,rho,alpha,c,d)
      data = [G,rho,alpha,c,d]
      data.append(Noise(target,noise_level))
      columns = ['G','rho','alpha','c','d','H_G']

      if(include_original_target):
         data.append(target)
         columns.append('H_G_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus7.0, Friedman Equation

      Arguments:
          G: float or array-like, default range (1.0,3.0)
          rho: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(8*pi*G*rho/3-alpha*c**2/d**2)
      """
      G = X[0]
      rho = X[1]
      alpha = X[2]
      c = X[3]
      d = X[4]
      return Bonus7.calculate(G,rho,alpha,c,d)

  @staticmethod
  def calculate(G,rho,alpha,c,d):
      """
      Bonus7.0, Friedman Equation

      Arguments:
          G: float or array-like, default range (1.0,3.0)
          rho: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          d: float or array-like, default range (1.0,3.0)
      Returns:
          f: sqrt(8*pi*G*rho/3-alpha*c**2/d**2)
      """
      return np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2)
  

class Bonus8:
  equation_lambda = lambda args : (lambda E_n,m,c,theta: E_n/(1+E_n/(m*c**2)*(1-np.cos(theta))) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus8.0, Compton Scattering

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','m','c','theta','K']
      """
      E_n = np.random.uniform(1.0,3.0, size)
      m = np.random.uniform(1.0,3.0, size)
      c = np.random.uniform(1.0,3.0, size)
      theta = np.random.uniform(1.0,3.0, size)
      return Bonus8.calculate_df(E_n,m,c,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(E_n,m,c,theta, noise_level = 0, include_original_target = False):
      """
      Bonus8.0, Compton Scattering

      Arguments:
          E_n: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['E_n','m','c','theta','K']
      """
      target = Bonus8.calculate(E_n,m,c,theta)
      data = [E_n,m,c,theta]
      data.append(Noise(target,noise_level))
      columns = ['E_n','m','c','theta','K']

      if(include_original_target):
         data.append(target)
         columns.append('K_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus8.0, Compton Scattering

      Arguments:
          E_n: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: E_n/(1+E_n/(m*c**2)*(1-cos(theta)))
      """
      E_n = X[0]
      m = X[1]
      c = X[2]
      theta = X[3]
      return Bonus8.calculate(E_n,m,c,theta)

  @staticmethod
  def calculate(E_n,m,c,theta):
      """
      Bonus8.0, Compton Scattering

      Arguments:
          E_n: float or array-like, default range (1.0,3.0)
          m: float or array-like, default range (1.0,3.0)
          c: float or array-like, default range (1.0,3.0)
          theta: float or array-like, default range (1.0,3.0)
      Returns:
          f: E_n/(1+E_n/(m*c**2)*(1-cos(theta)))
      """
      return E_n/(1+E_n/(m*c**2)*(1-np.cos(theta)))
  

class Bonus9:
  equation_lambda = lambda args : (lambda G,c,m1,m2,r: -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus9.0, Gravitational wave ratiated power

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','c','m1','m2','r','Pwr']
      """
      G = np.random.uniform(1.0,2.0, size)
      c = np.random.uniform(1.0,2.0, size)
      m1 = np.random.uniform(1.0,5.0, size)
      m2 = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,2.0, size)
      return Bonus9.calculate_df(G,c,m1,m2,r,noise_level,include_original_target)

  @staticmethod
  def calculate_df(G,c,m1,m2,r, noise_level = 0, include_original_target = False):
      """
      Bonus9.0, Gravitational wave ratiated power

      Arguments:
          G: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','c','m1','m2','r','Pwr']
      """
      target = Bonus9.calculate(G,c,m1,m2,r)
      data = [G,c,m1,m2,r]
      data.append(Noise(target,noise_level))
      columns = ['G','c','m1','m2','r','Pwr']

      if(include_original_target):
         data.append(target)
         columns.append('Pwr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus9.0, Gravitational wave ratiated power

      Arguments:
          G: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,2.0)
      Returns:
          f: -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
      """
      G = X[0]
      c = X[1]
      m1 = X[2]
      m2 = X[3]
      r = X[4]
      return Bonus9.calculate(G,c,m1,m2,r)

  @staticmethod
  def calculate(G,c,m1,m2,r):
      """
      Bonus9.0, Gravitational wave ratiated power

      Arguments:
          G: float or array-like, default range (1.0,2.0)
          c: float or array-like, default range (1.0,2.0)
          m1: float or array-like, default range (1.0,5.0)
          m2: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,2.0)
      Returns:
          f: -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
      """
      return -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
  

class Bonus10:
  equation_lambda = lambda args : (lambda c,v,theta2: np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2))) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus10.0, Relativistic aberation

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','theta2','theta1']
      """
      c = np.random.uniform(4.0,6.0, size)
      v = np.random.uniform(1.0,3.0, size)
      theta2 = np.random.uniform(1.0,3.0, size)
      return Bonus10.calculate_df(c,v,theta2,noise_level,include_original_target)

  @staticmethod
  def calculate_df(c,v,theta2, noise_level = 0, include_original_target = False):
      """
      Bonus10.0, Relativistic aberation

      Arguments:
          c: float or array-like, default range (4.0,6.0)
          v: float or array-like, default range (1.0,3.0)
          theta2: float or array-like, default range (1.0,3.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','theta2','theta1']
      """
      target = Bonus10.calculate(c,v,theta2)
      data = [c,v,theta2]
      data.append(Noise(target,noise_level))
      columns = ['c','v','theta2','theta1']

      if(include_original_target):
         data.append(target)
         columns.append('theta1_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus10.0, Relativistic aberation

      Arguments:
          c: float or array-like, default range (4.0,6.0)
          v: float or array-like, default range (1.0,3.0)
          theta2: float or array-like, default range (1.0,3.0)
      Returns:
          f: arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))
      """
      c = X[0]
      v = X[1]
      theta2 = X[2]
      return Bonus10.calculate(c,v,theta2)

  @staticmethod
  def calculate(c,v,theta2):
      """
      Bonus10.0, Relativistic aberation

      Arguments:
          c: float or array-like, default range (4.0,6.0)
          v: float or array-like, default range (1.0,3.0)
          theta2: float or array-like, default range (1.0,3.0)
      Returns:
          f: arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))
      """
      return np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2)))
  

class Bonus11:
  equation_lambda = lambda args : (lambda I_0,alpha,delta,n: I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2 )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus11.0, N-slit diffraction

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I_0','alpha','delta','n','I']
      """
      I_0 = np.random.uniform(1.0,3.0, size)
      alpha = np.random.uniform(1.0,3.0, size)
      delta = np.random.uniform(1.0,3.0, size)
      n = np.random.uniform(1.0,2.0, size)
      return Bonus11.calculate_df(I_0,alpha,delta,n,noise_level,include_original_target)

  @staticmethod
  def calculate_df(I_0,alpha,delta,n, noise_level = 0, include_original_target = False):
      """
      Bonus11.0, N-slit diffraction

      Arguments:
          I_0: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          delta: float or array-like, default range (1.0,3.0)
          n: float or array-like, default range (1.0,2.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['I_0','alpha','delta','n','I']
      """
      target = Bonus11.calculate(I_0,alpha,delta,n)
      data = [I_0,alpha,delta,n]
      data.append(Noise(target,noise_level))
      columns = ['I_0','alpha','delta','n','I']

      if(include_original_target):
         data.append(target)
         columns.append('I_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus11.0, N-slit diffraction

      Arguments:
          I_0: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          delta: float or array-like, default range (1.0,3.0)
          n: float or array-like, default range (1.0,2.0)
      Returns:
          f: I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2
      """
      I_0 = X[0]
      alpha = X[1]
      delta = X[2]
      n = X[3]
      return Bonus11.calculate(I_0,alpha,delta,n)

  @staticmethod
  def calculate(I_0,alpha,delta,n):
      """
      Bonus11.0, N-slit diffraction

      Arguments:
          I_0: float or array-like, default range (1.0,3.0)
          alpha: float or array-like, default range (1.0,3.0)
          delta: float or array-like, default range (1.0,3.0)
          n: float or array-like, default range (1.0,2.0)
      Returns:
          f: I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2
      """
      return I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2
  

class Bonus12:
  equation_lambda = lambda args : (lambda q,y,Volt,d,epsilon: q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus12.0, 2.11 Jackson

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','y','Volt','d','epsilon','F']
      """
      q = np.random.uniform(1.0,5.0, size)
      y = np.random.uniform(1.0,3.0, size)
      Volt = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(4.0,6.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      return Bonus12.calculate_df(q,y,Volt,d,epsilon,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,y,Volt,d,epsilon, noise_level = 0, include_original_target = False):
      """
      Bonus12.0, 2.11 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,3.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (4.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','y','Volt','d','epsilon','F']
      """
      target = Bonus12.calculate(q,y,Volt,d,epsilon)
      data = [q,y,Volt,d,epsilon]
      data.append(Noise(target,noise_level))
      columns = ['q','y','Volt','d','epsilon','F']

      if(include_original_target):
         data.append(target)
         columns.append('F_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus12.0, 2.11 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,3.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (4.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)
      """
      q = X[0]
      y = X[1]
      Volt = X[2]
      d = X[3]
      epsilon = X[4]
      return Bonus12.calculate(q,y,Volt,d,epsilon)

  @staticmethod
  def calculate(q,y,Volt,d,epsilon):
      """
      Bonus12.0, 2.11 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,3.0)
          Volt: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (4.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)
      """
      return q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)
  

class Bonus13:
  equation_lambda = lambda args : (lambda q,r,d,alpha,epsilon: 1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus13.0, 3.45 Jackson

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','r','d','alpha','epsilon','Volt']
      """
      q = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,3.0, size)
      d = np.random.uniform(4.0,6.0, size)
      alpha = np.random.uniform(0.0,6.0, size)
      epsilon = np.random.uniform(1.0,5.0, size)
      return Bonus13.calculate_df(q,r,d,alpha,epsilon,noise_level,include_original_target)

  @staticmethod
  def calculate_df(q,r,d,alpha,epsilon, noise_level = 0, include_original_target = False):
      """
      Bonus13.0, 3.45 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,3.0)
          d: float or array-like, default range (4.0,6.0)
          alpha: float or array-like, default range (0.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['q','r','d','alpha','epsilon','Volt']
      """
      target = Bonus13.calculate(q,r,d,alpha,epsilon)
      data = [q,r,d,alpha,epsilon]
      data.append(Noise(target,noise_level))
      columns = ['q','r','d','alpha','epsilon','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus13.0, 3.45 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,3.0)
          d: float or array-like, default range (4.0,6.0)
          alpha: float or array-like, default range (0.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))
      """
      q = X[0]
      r = X[1]
      d = X[2]
      alpha = X[3]
      epsilon = X[4]
      return Bonus13.calculate(q,r,d,alpha,epsilon)

  @staticmethod
  def calculate(q,r,d,alpha,epsilon):
      """
      Bonus13.0, 3.45 Jackson

      Arguments:
          q: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,3.0)
          d: float or array-like, default range (4.0,6.0)
          alpha: float or array-like, default range (0.0,6.0)
          epsilon: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))
      """
      return 1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha))
  

class Bonus14:
  equation_lambda = lambda args : (lambda Ef,theta,r,d,alpha: Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus14.0, 4.60' Jackson

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Ef','theta','r','d','alpha','Volt']
      """
      Ef = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(0.0,6.0, size)
      r = np.random.uniform(1.0,5.0, size)
      d = np.random.uniform(1.0,5.0, size)
      alpha = np.random.uniform(1.0,5.0, size)
      return Bonus14.calculate_df(Ef,theta,r,d,alpha,noise_level,include_original_target)

  @staticmethod
  def calculate_df(Ef,theta,r,d,alpha, noise_level = 0, include_original_target = False):
      """
      Bonus14.0, 4.60' Jackson

      Arguments:
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
          r: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['Ef','theta','r','d','alpha','Volt']
      """
      target = Bonus14.calculate(Ef,theta,r,d,alpha)
      data = [Ef,theta,r,d,alpha]
      data.append(Noise(target,noise_level))
      columns = ['Ef','theta','r','d','alpha','Volt']

      if(include_original_target):
         data.append(target)
         columns.append('Volt_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus14.0, 4.60' Jackson

      Arguments:
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
          r: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
      Returns:
          f: Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))
      """
      Ef = X[0]
      theta = X[1]
      r = X[2]
      d = X[3]
      alpha = X[4]
      return Bonus14.calculate(Ef,theta,r,d,alpha)

  @staticmethod
  def calculate(Ef,theta,r,d,alpha):
      """
      Bonus14.0, 4.60' Jackson

      Arguments:
          Ef: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
          r: float or array-like, default range (1.0,5.0)
          d: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
      Returns:
          f: Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))
      """
      return Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))
  

class Bonus15:
  equation_lambda = lambda args : (lambda c,v,omega,theta: np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus15.0, 11.38 Jackson

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega','theta','omega_0']
      """
      c = np.random.uniform(5.0,20.0, size)
      v = np.random.uniform(1.0,3.0, size)
      omega = np.random.uniform(1.0,5.0, size)
      theta = np.random.uniform(0.0,6.0, size)
      return Bonus15.calculate_df(c,v,omega,theta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(c,v,omega,theta, noise_level = 0, include_original_target = False):
      """
      Bonus15.0, 11.38 Jackson

      Arguments:
          c: float or array-like, default range (5.0,20.0)
          v: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['c','v','omega','theta','omega_0']
      """
      target = Bonus15.calculate(c,v,omega,theta)
      data = [c,v,omega,theta]
      data.append(Noise(target,noise_level))
      columns = ['c','v','omega','theta','omega_0']

      if(include_original_target):
         data.append(target)
         columns.append('omega_0_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus15.0, 11.38 Jackson

      Arguments:
          c: float or array-like, default range (5.0,20.0)
          v: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
      Returns:
          f: sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))
      """
      c = X[0]
      v = X[1]
      omega = X[2]
      theta = X[3]
      return Bonus15.calculate(c,v,omega,theta)

  @staticmethod
  def calculate(c,v,omega,theta):
      """
      Bonus15.0, 11.38 Jackson

      Arguments:
          c: float or array-like, default range (5.0,20.0)
          v: float or array-like, default range (1.0,3.0)
          omega: float or array-like, default range (1.0,5.0)
          theta: float or array-like, default range (0.0,6.0)
      Returns:
          f: sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))
      """
      return np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta))
  

class Bonus16:
  equation_lambda = lambda args : (lambda m,c,p,q,A_vec,Volt: np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus16.0, 8.56 Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','c','p','q','A_vec','Volt','E_n']
      """
      m = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      p = np.random.uniform(1.0,5.0, size)
      q = np.random.uniform(1.0,5.0, size)
      A_vec = np.random.uniform(1.0,5.0, size)
      Volt = np.random.uniform(1.0,5.0, size)
      return Bonus16.calculate_df(m,c,p,q,A_vec,Volt,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,c,p,q,A_vec,Volt, noise_level = 0, include_original_target = False):
      """
      Bonus16.0, 8.56 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','c','p','q','A_vec','Volt','E_n']
      """
      target = Bonus16.calculate(m,c,p,q,A_vec,Volt)
      data = [m,c,p,q,A_vec,Volt]
      data.append(Noise(target,noise_level))
      columns = ['m','c','p','q','A_vec','Volt','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus16.0, 8.56 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt
      """
      m = X[0]
      c = X[1]
      p = X[2]
      q = X[3]
      A_vec = X[4]
      Volt = X[5]
      return Bonus16.calculate(m,c,p,q,A_vec,Volt)

  @staticmethod
  def calculate(m,c,p,q,A_vec,Volt):
      """
      Bonus16.0, 8.56 Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          q: float or array-like, default range (1.0,5.0)
          A_vec: float or array-like, default range (1.0,5.0)
          Volt: float or array-like, default range (1.0,5.0)
      Returns:
          f: sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt
      """
      return np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt
  

class Bonus17:
  equation_lambda = lambda args : (lambda m,omega,p,y,x,alpha: 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus17.0, 12.80' Goldstein

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','omega','p','y','x','alpha','E_n']
      """
      m = np.random.uniform(1.0,5.0, size)
      omega = np.random.uniform(1.0,5.0, size)
      p = np.random.uniform(1.0,5.0, size)
      y = np.random.uniform(1.0,5.0, size)
      x = np.random.uniform(1.0,5.0, size)
      alpha = np.random.uniform(1.0,5.0, size)
      return Bonus17.calculate_df(m,omega,p,y,x,alpha,noise_level,include_original_target)

  @staticmethod
  def calculate_df(m,omega,p,y,x,alpha, noise_level = 0, include_original_target = False):
      """
      Bonus17.0, 12.80' Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['m','omega','p','y','x','alpha','E_n']
      """
      target = Bonus17.calculate(m,omega,p,y,x,alpha)
      data = [m,omega,p,y,x,alpha]
      data.append(Noise(target,noise_level))
      columns = ['m','omega','p','y','x','alpha','E_n']

      if(include_original_target):
         data.append(target)
         columns.append('E_n_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus17.0, 12.80' Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))
      """
      m = X[0]
      omega = X[1]
      p = X[2]
      y = X[3]
      x = X[4]
      alpha = X[5]
      return Bonus17.calculate(m,omega,p,y,x,alpha)

  @staticmethod
  def calculate(m,omega,p,y,x,alpha):
      """
      Bonus17.0, 12.80' Goldstein

      Arguments:
          m: float or array-like, default range (1.0,5.0)
          omega: float or array-like, default range (1.0,5.0)
          p: float or array-like, default range (1.0,5.0)
          y: float or array-like, default range (1.0,5.0)
          x: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
      Returns:
          f: 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))
      """
      return 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))
  

class Bonus18:
  equation_lambda = lambda args : (lambda G,k_f,r,H_G,c: 3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus18.0, 15.2.1 Weinberg

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','k_f','r','H_G','c','rho_0']
      """
      G = np.random.uniform(1.0,5.0, size)
      k_f = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      H_G = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      return Bonus18.calculate_df(G,k_f,r,H_G,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(G,k_f,r,H_G,c, noise_level = 0, include_original_target = False):
      """
      Bonus18.0, 15.2.1 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','k_f','r','H_G','c','rho_0']
      """
      target = Bonus18.calculate(G,k_f,r,H_G,c)
      data = [G,k_f,r,H_G,c]
      data.append(Noise(target,noise_level))
      columns = ['G','k_f','r','H_G','c','rho_0']

      if(include_original_target):
         data.append(target)
         columns.append('rho_0_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus18.0, 15.2.1 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)
      """
      G = X[0]
      k_f = X[1]
      r = X[2]
      H_G = X[3]
      c = X[4]
      return Bonus18.calculate(G,k_f,r,H_G,c)

  @staticmethod
  def calculate(G,k_f,r,H_G,c):
      """
      Bonus18.0, 15.2.1 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: 3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)
      """
      return 3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2)
  

class Bonus19:
  equation_lambda = lambda args : (lambda G,k_f,r,H_G,alpha,c: -1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha)) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus19.0, 15.2.2 Weinberg

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','k_f','r','H_G','alpha','c','pr']
      """
      G = np.random.uniform(1.0,5.0, size)
      k_f = np.random.uniform(1.0,5.0, size)
      r = np.random.uniform(1.0,5.0, size)
      H_G = np.random.uniform(1.0,5.0, size)
      alpha = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      return Bonus19.calculate_df(G,k_f,r,H_G,alpha,c,noise_level,include_original_target)

  @staticmethod
  def calculate_df(G,k_f,r,H_G,alpha,c, noise_level = 0, include_original_target = False):
      """
      Bonus19.0, 15.2.2 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['G','k_f','r','H_G','alpha','c','pr']
      """
      target = Bonus19.calculate(G,k_f,r,H_G,alpha,c)
      data = [G,k_f,r,H_G,alpha,c]
      data.append(Noise(target,noise_level))
      columns = ['G','k_f','r','H_G','alpha','c','pr']

      if(include_original_target):
         data.append(target)
         columns.append('pr_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus19.0, 15.2.2 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: -1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))
      """
      G = X[0]
      k_f = X[1]
      r = X[2]
      H_G = X[3]
      alpha = X[4]
      c = X[5]
      return Bonus19.calculate(G,k_f,r,H_G,alpha,c)

  @staticmethod
  def calculate(G,k_f,r,H_G,alpha,c):
      """
      Bonus19.0, 15.2.2 Weinberg

      Arguments:
          G: float or array-like, default range (1.0,5.0)
          k_f: float or array-like, default range (1.0,5.0)
          r: float or array-like, default range (1.0,5.0)
          H_G: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
      Returns:
          f: -1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))
      """
      return -1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))
  

class Bonus20:
  equation_lambda = lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2) )(*args)

  @staticmethod
  def generate_df(size = 10000, noise_level = 0, include_original_target = False):
      """
      Bonus20.0, Klein-Nishina (13.132 Schwarz)

      Arguments:
          size: length of the inputs,
                sampled from the uniformly distributed standard ranges
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','omega_0','alpha','h','m','c','beta','A']
      """
      omega = np.random.uniform(1.0,5.0, size)
      omega_0 = np.random.uniform(1.0,5.0, size)
      alpha = np.random.uniform(1.0,5.0, size)
      h = np.random.uniform(1.0,5.0, size)
      m = np.random.uniform(1.0,5.0, size)
      c = np.random.uniform(1.0,5.0, size)
      beta = np.random.uniform(0.0,6.0, size)
      return Bonus20.calculate_df(omega,omega_0,alpha,h,m,c,beta,noise_level,include_original_target)

  @staticmethod
  def calculate_df(omega,omega_0,alpha,h,m,c,beta, noise_level = 0, include_original_target = False):
      """
      Bonus20.0, Klein-Nishina (13.132 Schwarz)

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          beta: float or array-like, default range (0.0,6.0)
          noise_level: normal distributed noise added as target's 
                standard deviation times sqrt(noise_level/(1-noise_level))
      Returns:
          pandas DataFrame ['omega','omega_0','alpha','h','m','c','beta','A']
      """
      target = Bonus20.calculate(omega,omega_0,alpha,h,m,c,beta)
      data = [omega,omega_0,alpha,h,m,c,beta]
      data.append(Noise(target,noise_level))
      columns = ['omega','omega_0','alpha','h','m','c','beta','A']

      if(include_original_target):
         data.append(target)
         columns.append('A_without_noise')
      return pd.DataFrame( list(zip(*data)), columns=columns)

  @staticmethod
  def calculate_batch(X):
      """
      Bonus20.0, Klein-Nishina (13.132 Schwarz)

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          beta: float or array-like, default range (0.0,6.0)
      Returns:
          f: 1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)
      """
      omega = X[0]
      omega_0 = X[1]
      alpha = X[2]
      h = X[3]
      m = X[4]
      c = X[5]
      beta = X[6]
      return Bonus20.calculate(omega,omega_0,alpha,h,m,c,beta)

  @staticmethod
  def calculate(omega,omega_0,alpha,h,m,c,beta):
      """
      Bonus20.0, Klein-Nishina (13.132 Schwarz)

      Arguments:
          omega: float or array-like, default range (1.0,5.0)
          omega_0: float or array-like, default range (1.0,5.0)
          alpha: float or array-like, default range (1.0,5.0)
          h: float or array-like, default range (1.0,5.0)
          m: float or array-like, default range (1.0,5.0)
          c: float or array-like, default range (1.0,5.0)
          beta: float or array-like, default range (0.0,6.0)
      Returns:
          f: 1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)
      """
      return 1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2)
  
FunctionsJson = [
{'EquationName': 'Feynman1', 'DescriptiveName': 'Feynman1, Lecture I.6.2a', 'Formula_Str': 'exp(-theta**2/2)/sqrt(2*pi)', 'Formula': 'np.exp(-theta**2/2)/np.sqrt(2*np.pi)', 'Formula_Lambda': 'lambda args : (lambda theta: np.exp(-theta**2/2)/np.sqrt(2*np.pi) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda theta: {0} )(*args)', 'Variables': [{'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman2', 'DescriptiveName': 'Feynman2, Lecture I.6.2', 'Formula_Str': 'exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)', 'Formula': 'np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)', 'Formula_Lambda': 'lambda args : (lambda sigma,theta: np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda sigma,theta: {0} )(*args)', 'Variables': [{'name': 'sigma', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman3', 'DescriptiveName': 'Feynman3, Lecture I.6.2b', 'Formula_Str': 'exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)', 'Formula': 'np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)', 'Formula_Lambda': 'lambda args : (lambda sigma,theta,theta1: np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda sigma,theta,theta1: {0} )(*args)', 'Variables': [{'name': 'sigma', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'theta1', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman4', 'DescriptiveName': 'Feynman4, Lecture I.8.14', 'Formula_Str': 'sqrt((x2-x1)**2+(y2-y1)**2)', 'Formula': 'np.sqrt((x2-x1)**2+(y2-y1)**2)', 'Formula_Lambda': 'lambda args : (lambda x1,x2,y1,y2: np.sqrt((x2-x1)**2+(y2-y1)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x1,x2,y1,y2: {0} )(*args)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'y1', 'low': 1.0, 'high': 5.0}, {'name': 'y2', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman5', 'DescriptiveName': 'Feynman5, Lecture I.9.18', 'Formula_Str': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)', 'Formula': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)', 'Formula_Lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: {0} )(*args)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 2.0}, {'name': 'm2', 'low': 1.0, 'high': 2.0}, {'name': 'G', 'low': 1.0, 'high': 2.0}, {'name': 'x1', 'low': 3.0, 'high': 4.0}, {'name': 'x2', 'low': 1.0, 'high': 2.0}, {'name': 'y1', 'low': 3.0, 'high': 4.0}, {'name': 'y2', 'low': 1.0, 'high': 2.0}, {'name': 'z1', 'low': 3.0, 'high': 4.0}, {'name': 'z2', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Feynman6', 'DescriptiveName': 'Feynman6, Lecture I.10.7', 'Formula_Str': 'm_0/sqrt(1-v**2/c**2)', 'Formula': 'm_0/np.sqrt(1-v**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda m_0,v,c: m_0/np.sqrt(1-v**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m_0,v,c: {0} )(*args)', 'Variables': [{'name': 'm_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman7', 'DescriptiveName': 'Feynman7, Lecture I.11.19', 'Formula_Str': 'x1*y1+x2*y2+x3*y3', 'Formula': 'x1*y1+x2*y2+x3*y3', 'Formula_Lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: x1*y1+x2*y2+x3*y3 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: {0} )(*args)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'x3', 'low': 1.0, 'high': 5.0}, {'name': 'y1', 'low': 1.0, 'high': 5.0}, {'name': 'y2', 'low': 1.0, 'high': 5.0}, {'name': 'y3', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman8', 'DescriptiveName': 'Feynman8, Lecture I.12.1', 'Formula_Str': 'mu*Nn', 'Formula': 'mu*Nn', 'Formula_Lambda': 'lambda args : (lambda mu,Nn: mu*Nn )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mu,Nn: {0} )(*args)', 'Variables': [{'name': 'mu', 'low': 1.0, 'high': 5.0}, {'name': 'Nn', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman10', 'DescriptiveName': 'Feynman10, Lecture I.12.2', 'Formula_Str': 'q1*q2*r/(4*pi*epsilon*r**3)', 'Formula': 'q1*q2*r/(4*np.pi*epsilon*r**3)', 'Formula_Lambda': 'lambda args : (lambda q1,q2,epsilon,r: q1*q2*r/(4*np.pi*epsilon*r**3) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q1,q2,epsilon,r: {0} )(*args)', 'Variables': [{'name': 'q1', 'low': 1.0, 'high': 5.0}, {'name': 'q2', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman11', 'DescriptiveName': 'Feynman11, Lecture I.12.4', 'Formula_Str': 'q1*r/(4*pi*epsilon*r**3)', 'Formula': 'q1*r/(4*np.pi*epsilon*r**3)', 'Formula_Lambda': 'lambda args : (lambda q1,epsilon,r: q1*r/(4*np.pi*epsilon*r**3) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q1,epsilon,r: {0} )(*args)', 'Variables': [{'name': 'q1', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman12', 'DescriptiveName': 'Feynman12, Lecture I.12.5', 'Formula_Str': 'q2*Ef', 'Formula': 'q2*Ef', 'Formula_Lambda': 'lambda args : (lambda q2,Ef: q2*Ef )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q2,Ef: {0} )(*args)', 'Variables': [{'name': 'q2', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman13', 'DescriptiveName': 'Feynman13, Lecture I.12.11', 'Formula_Str': 'q*(Ef+B*v*sin(theta))', 'Formula': 'q*(Ef+B*v*np.sin(theta))', 'Formula_Lambda': 'lambda args : (lambda q,Ef,B,v,theta: q*(Ef+B*v*np.sin(theta)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,Ef,B,v,theta: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman9', 'DescriptiveName': 'Feynman9, Lecture I.13.4', 'Formula_Str': '1/2*m*(v**2+u**2+w**2)', 'Formula': '1/2*m*(v**2+u**2+w**2)', 'Formula_Lambda': 'lambda args : (lambda m,v,u,w: 1/2*m*(v**2+u**2+w**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,v,u,w: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'u', 'low': 1.0, 'high': 5.0}, {'name': 'w', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman14', 'DescriptiveName': 'Feynman14, Lecture I.13.12', 'Formula_Str': 'G*m1*m2*(1/r2-1/r1)', 'Formula': 'G*m1*m2*(1/r2-1/r1)', 'Formula_Lambda': 'lambda args : (lambda m1,m2,r1,r2,G: G*m1*m2*(1/r2-1/r1) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m1,m2,r1,r2,G: {0} )(*args)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r1', 'low': 1.0, 'high': 5.0}, {'name': 'r2', 'low': 1.0, 'high': 5.0}, {'name': 'G', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman15', 'DescriptiveName': 'Feynman15, Lecture I.14.3', 'Formula_Str': 'm*g*z', 'Formula': 'm*g*z', 'Formula_Lambda': 'lambda args : (lambda m,g,z: m*g*z )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,g,z: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'g', 'low': 1.0, 'high': 5.0}, {'name': 'z', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman16', 'DescriptiveName': 'Feynman16, Lecture I.14.4', 'Formula_Str': '1/2*k_spring*x**2', 'Formula': '1/2*k_spring*x**2', 'Formula_Lambda': 'lambda args : (lambda k_spring,x: 1/2*k_spring*x**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda k_spring,x: {0} )(*args)', 'Variables': [{'name': 'k_spring', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman17', 'DescriptiveName': 'Feynman17, Lecture I.15.3x', 'Formula_Str': '(x-u*t)/sqrt(1-u**2/c**2)', 'Formula': '(x-u*t)/np.sqrt(1-u**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda x,u,c,t: (x-u*t)/np.sqrt(1-u**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x,u,c,t: {0} )(*args)', 'Variables': [{'name': 'x', 'low': 5.0, 'high': 10.0}, {'name': 'u', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 20.0}, {'name': 't', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Feynman18', 'DescriptiveName': 'Feynman18, Lecture I.15.3t', 'Formula_Str': '(t-u*x/c**2)/sqrt(1-u**2/c**2)', 'Formula': '(t-u*x/c**2)/np.sqrt(1-u**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda x,c,u,t: (t-u*x/c**2)/np.sqrt(1-u**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x,c,u,t: {0} )(*args)', 'Variables': [{'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'u', 'low': 1.0, 'high': 2.0}, {'name': 't', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman19', 'DescriptiveName': 'Feynman19, Lecture I.15.1', 'Formula_Str': 'm_0*v/sqrt(1-v**2/c**2)', 'Formula': 'm_0*v/np.sqrt(1-v**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda m_0,v,c: m_0*v/np.sqrt(1-v**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m_0,v,c: {0} )(*args)', 'Variables': [{'name': 'm_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman20', 'DescriptiveName': 'Feynman20, Lecture I.16.6', 'Formula_Str': '(u+v)/(1+u*v/c**2)', 'Formula': '(u+v)/(1+u*v/c**2)', 'Formula_Lambda': 'lambda args : (lambda c,v,u: (u+v)/(1+u*v/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda c,v,u: {0} )(*args)', 'Variables': [{'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'u', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman21', 'DescriptiveName': 'Feynman21, Lecture I.18.4', 'Formula_Str': '(m1*r1+m2*r2)/(m1+m2)', 'Formula': '(m1*r1+m2*r2)/(m1+m2)', 'Formula_Lambda': 'lambda args : (lambda m1,m2,r1,r2: (m1*r1+m2*r2)/(m1+m2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m1,m2,r1,r2: {0} )(*args)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r1', 'low': 1.0, 'high': 5.0}, {'name': 'r2', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman22', 'DescriptiveName': 'Feynman22, Lecture I.18.12', 'Formula_Str': 'r*F*sin(theta)', 'Formula': 'r*F*np.sin(theta)', 'Formula_Lambda': 'lambda args : (lambda r,F,theta: r*F*np.sin(theta) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda r,F,theta: {0} )(*args)', 'Variables': [{'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'F', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 5.0}]},
{'EquationName': 'Feynman23', 'DescriptiveName': 'Feynman23, Lecture I.18.14', 'Formula_Str': 'm*r*v*sin(theta)', 'Formula': 'm*r*v*np.sin(theta)', 'Formula_Lambda': 'lambda args : (lambda m,r,v,theta: m*r*v*np.sin(theta) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,r,v,theta: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman24', 'DescriptiveName': 'Feynman24, Lecture I.24.6', 'Formula_Str': '1/2*m*(omega**2+omega_0**2)*1/2*x**2', 'Formula': '1/2*m*(omega**2+omega_0**2)*1/2*x**2', 'Formula_Lambda': 'lambda args : (lambda m,omega,omega_0,x: 1/2*m*(omega**2+omega_0**2)*1/2*x**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,omega,omega_0,x: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 3.0}, {'name': 'omega_0', 'low': 1.0, 'high': 3.0}, {'name': 'x', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman25', 'DescriptiveName': 'Feynman25, Lecture I.25.13', 'Formula_Str': 'q/C', 'Formula': 'q/C', 'Formula_Lambda': 'lambda args : (lambda q,C: q/C )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,C: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'C', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman26', 'DescriptiveName': 'Feynman26, Lecture I.26.2', 'Formula_Str': 'arcsin(n*sin(theta2))', 'Formula': 'np.arcsin(n*np.sin(theta2))', 'Formula_Lambda': 'lambda args : (lambda n,theta2: np.arcsin(n*np.sin(theta2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,theta2: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'theta2', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman27', 'DescriptiveName': 'Feynman27, Lecture I.27.6', 'Formula_Str': '1/(1/d1+n/d2)', 'Formula': '1/(1/d1+n/d2)', 'Formula_Lambda': 'lambda args : (lambda d1,d2,n: 1/(1/d1+n/d2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda d1,d2,n: {0} )(*args)', 'Variables': [{'name': 'd1', 'low': 1.0, 'high': 5.0}, {'name': 'd2', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman28', 'DescriptiveName': 'Feynman28, Lecture I.29.4', 'Formula_Str': 'omega/c', 'Formula': 'omega/c', 'Formula_Lambda': 'lambda args : (lambda omega,c: omega/c )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda omega,c: {0} )(*args)', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 10.0}, {'name': 'c', 'low': 1.0, 'high': 10.0}]},
{'EquationName': 'Feynman29', 'DescriptiveName': 'Feynman29, Lecture I.29.16', 'Formula_Str': 'sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))', 'Formula': 'np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))', 'Formula_Lambda': 'lambda args : (lambda x1,x2,theta1,theta2: np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x1,x2,theta1,theta2: {0} )(*args)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'theta1', 'low': 1.0, 'high': 5.0}, {'name': 'theta2', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman30', 'DescriptiveName': 'Feynman30, Lecture I.30.3', 'Formula_Str': 'Int_0*sin(n*theta/2)**2/sin(theta/2)**2', 'Formula': 'Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2', 'Formula_Lambda': 'lambda args : (lambda Int_0,theta,n: Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Int_0,theta,n: {0} )(*args)', 'Variables': [{'name': 'Int_0', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman31', 'DescriptiveName': 'Feynman31, Lecture I.30.5', 'Formula_Str': 'arcsin(lambd/(n*d))', 'Formula': 'np.arcsin(lambd/(n*d))', 'Formula_Lambda': 'lambda args : (lambda lambd,d,n: np.arcsin(lambd/(n*d)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda lambd,d,n: {0} )(*args)', 'Variables': [{'name': 'lambd', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 2.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman32', 'DescriptiveName': 'Feynman32, Lecture I.32.5', 'Formula_Str': 'q**2*a**2/(6*pi*epsilon*c**3)', 'Formula': 'q**2*a**2/(6*np.pi*epsilon*c**3)', 'Formula_Lambda': 'lambda args : (lambda q,a,epsilon,c: q**2*a**2/(6*np.pi*epsilon*c**3) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,a,epsilon,c: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'a', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman33', 'DescriptiveName': 'Feynman33, Lecture I.32.17', 'Formula_Str': '(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)', 'Formula': '(1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)', 'Formula_Lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'Ef', 'low': 1.0, 'high': 2.0}, {'name': 'r', 'low': 1.0, 'high': 2.0}, {'name': 'omega', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 3.0, 'high': 5.0}]},
{'EquationName': 'Feynman34', 'DescriptiveName': 'Feynman34, Lecture I.34.8', 'Formula_Str': 'q*v*B/p', 'Formula': 'q*v*B/p', 'Formula_Lambda': 'lambda args : (lambda q,v,B,p: q*v*B/p )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,v,B,p: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman35', 'DescriptiveName': 'Feynman35, Lecture I.34.1', 'Formula_Str': 'omega_0/(1-v/c)', 'Formula': 'omega_0/(1-v/c)', 'Formula_Lambda': 'lambda args : (lambda c,v,omega_0: omega_0/(1-v/c) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda c,v,omega_0: {0} )(*args)', 'Variables': [{'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman36', 'DescriptiveName': 'Feynman36, Lecture I.34.14', 'Formula_Str': '(1+v/c)/sqrt(1-v**2/c**2)*omega_0', 'Formula': '(1+v/c)/np.sqrt(1-v**2/c**2)*omega_0', 'Formula_Lambda': 'lambda args : (lambda c,v,omega_0: (1+v/c)/np.sqrt(1-v**2/c**2)*omega_0 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda c,v,omega_0: {0} )(*args)', 'Variables': [{'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman37', 'DescriptiveName': 'Feynman37, Lecture I.34.27', 'Formula_Str': '(h/(2*pi))*omega', 'Formula': '(h/(2*np.pi))*omega', 'Formula_Lambda': 'lambda args : (lambda omega,h: (h/(2*np.pi))*omega )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda omega,h: {0} )(*args)', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman38', 'DescriptiveName': 'Feynman38, Lecture I.37.4', 'Formula_Str': 'I1+I2+2*sqrt(I1*I2)*cos(delta)', 'Formula': 'I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)', 'Formula_Lambda': 'lambda args : (lambda I1,I2,delta: I1+I2+2*np.sqrt(I1*I2)*np.cos(delta) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda I1,I2,delta: {0} )(*args)', 'Variables': [{'name': 'I1', 'low': 1.0, 'high': 5.0}, {'name': 'I2', 'low': 1.0, 'high': 5.0}, {'name': 'delta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman39', 'DescriptiveName': 'Feynman39, Lecture I.38.12', 'Formula_Str': '4*pi*epsilon*(h/(2*pi))**2/(m*q**2)', 'Formula': '4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)', 'Formula_Lambda': 'lambda args : (lambda m,q,h,epsilon: 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,q,h,epsilon: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman40', 'DescriptiveName': 'Feynman40, Lecture I.39.1', 'Formula_Str': '3/2*pr*V', 'Formula': '3/2*pr*V', 'Formula_Lambda': 'lambda args : (lambda pr,V: 3/2*pr*V )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda pr,V: {0} )(*args)', 'Variables': [{'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman41', 'DescriptiveName': 'Feynman41, Lecture I.39.11', 'Formula_Str': '1/(gamma-1)*pr*V', 'Formula': '1/(gamma-1)*pr*V', 'Formula_Lambda': 'lambda args : (lambda gamma,pr,V: 1/(gamma-1)*pr*V )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda gamma,pr,V: {0} )(*args)', 'Variables': [{'name': 'gamma', 'low': 2.0, 'high': 5.0}, {'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman42', 'DescriptiveName': 'Feynman42, Lecture I.39.22', 'Formula_Str': 'n*kb*T/V', 'Formula': 'n*kb*T/V', 'Formula_Lambda': 'lambda args : (lambda n,T,V,kb: n*kb*T/V )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,T,V,kb: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman43', 'DescriptiveName': 'Feynman43, Lecture I.40.1', 'Formula_Str': 'n_0*exp(-m*g*x/(kb*T))', 'Formula': 'n_0*np.exp(-m*g*x/(kb*T))', 'Formula_Lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: n_0*np.exp(-m*g*x/(kb*T)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n_0,m,x,T,g,kb: {0} )(*args)', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'g', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman44', 'DescriptiveName': 'Feynman44, Lecture I.41.16', 'Formula_Str': 'h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))', 'Formula': 'h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))', 'Formula_Lambda': 'lambda args : (lambda omega,T,h,kb,c: h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda omega,T,h,kb,c: {0} )(*args)', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman45', 'DescriptiveName': 'Feynman45, Lecture I.43.16', 'Formula_Str': 'mu_drift*q*Volt/d', 'Formula': 'mu_drift*q*Volt/d', 'Formula_Lambda': 'lambda args : (lambda mu_drift,q,Volt,d: mu_drift*q*Volt/d )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mu_drift,q,Volt,d: {0} )(*args)', 'Variables': [{'name': 'mu_drift', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman46', 'DescriptiveName': 'Feynman46, Lecture I.43.31', 'Formula_Str': 'mob*kb*T', 'Formula': 'mob*kb*T', 'Formula_Lambda': 'lambda args : (lambda mob,T,kb: mob*kb*T )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mob,T,kb: {0} )(*args)', 'Variables': [{'name': 'mob', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman47', 'DescriptiveName': 'Feynman47, Lecture I.43.43', 'Formula_Str': '1/(gamma-1)*kb*v/A', 'Formula': '1/(gamma-1)*kb*v/A', 'Formula_Lambda': 'lambda args : (lambda gamma,kb,A,v: 1/(gamma-1)*kb*v/A )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda gamma,kb,A,v: {0} )(*args)', 'Variables': [{'name': 'gamma', 'low': 2.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman48', 'DescriptiveName': 'Feynman48, Lecture I.44.4', 'Formula_Str': 'n*kb*T*ln(V2/V1)', 'Formula': 'n*kb*T*np.ln(V2/V1)', 'Formula_Lambda': 'lambda args : (lambda n,kb,T,V1,V2: n*kb*T*np.ln(V2/V1) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,kb,T,V1,V2: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'V1', 'low': 1.0, 'high': 5.0}, {'name': 'V2', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman49', 'DescriptiveName': 'Feynman49, Lecture I.47.23', 'Formula_Str': 'sqrt(gamma*pr/rho)', 'Formula': 'np.sqrt(gamma*pr/rho)', 'Formula_Lambda': 'lambda args : (lambda gamma,pr,rho: np.sqrt(gamma*pr/rho) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda gamma,pr,rho: {0} )(*args)', 'Variables': [{'name': 'gamma', 'low': 1.0, 'high': 5.0}, {'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'rho', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman50', 'DescriptiveName': 'Feynman50, Lecture I.48.2', 'Formula_Str': 'm*c**2/sqrt(1-v**2/c**2)', 'Formula': 'm*c**2/np.sqrt(1-v**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda m,v,c: m*c**2/np.sqrt(1-v**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,v,c: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman51', 'DescriptiveName': 'Feynman51, Lecture I.50.26', 'Formula_Str': 'x1*(cos(omega*t)+alpha*cos(omega*t)**2)', 'Formula': 'x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)', 'Formula_Lambda': 'lambda args : (lambda x1,omega,t,alpha: x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda x1,omega,t,alpha: {0} )(*args)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 3.0}, {'name': 't', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman52', 'DescriptiveName': 'Feynman52, Lecture II.2.42', 'Formula_Str': 'kappa*(T2-T1)*A/d', 'Formula': 'kappa*(T2-T1)*A/d', 'Formula_Lambda': 'lambda args : (lambda kappa,T1,T2,A,d: kappa*(T2-T1)*A/d )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda kappa,T1,T2,A,d: {0} )(*args)', 'Variables': [{'name': 'kappa', 'low': 1.0, 'high': 5.0}, {'name': 'T1', 'low': 1.0, 'high': 5.0}, {'name': 'T2', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman53', 'DescriptiveName': 'Feynman53, Lecture II.3.24', 'Formula_Str': 'Pwr/(4*pi*r**2)', 'Formula': 'Pwr/(4*np.pi*r**2)', 'Formula_Lambda': 'lambda args : (lambda Pwr,r: Pwr/(4*np.pi*r**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Pwr,r: {0} )(*args)', 'Variables': [{'name': 'Pwr', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman54', 'DescriptiveName': 'Feynman54, Lecture II.4.23', 'Formula_Str': 'q/(4*pi*epsilon*r)', 'Formula': 'q/(4*np.pi*epsilon*r)', 'Formula_Lambda': 'lambda args : (lambda q,epsilon,r: q/(4*np.pi*epsilon*r) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,epsilon,r: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman55', 'DescriptiveName': 'Feynman55, Lecture II.6.11', 'Formula_Str': '1/(4*pi*epsilon)*p_d*cos(theta)/r**2', 'Formula': '1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2', 'Formula_Lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,p_d,theta,r: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman56', 'DescriptiveName': 'Feynman56, Lecture II.6.15a', 'Formula_Str': 'p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)', 'Formula': 'p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)', 'Formula_Lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,p_d,r,x,y,z: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}, {'name': 'x', 'low': 1.0, 'high': 3.0}, {'name': 'y', 'low': 1.0, 'high': 3.0}, {'name': 'z', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman57', 'DescriptiveName': 'Feynman57, Lecture II.6.15b', 'Formula_Str': 'p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3', 'Formula': 'p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3', 'Formula_Lambda': 'lambda args : (lambda epsilon,p_d,theta,r: p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,p_d,theta,r: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman58', 'DescriptiveName': 'Feynman58, Lecture II.8.7', 'Formula_Str': '3/5*q**2/(4*pi*epsilon*d)', 'Formula': '3/5*q**2/(4*np.pi*epsilon*d)', 'Formula_Lambda': 'lambda args : (lambda q,epsilon,d: 3/5*q**2/(4*np.pi*epsilon*d) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,epsilon,d: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman59', 'DescriptiveName': 'Feynman59, Lecture II.8.31', 'Formula_Str': 'epsilon*Ef**2/2', 'Formula': 'epsilon*Ef**2/2', 'Formula_Lambda': 'lambda args : (lambda epsilon,Ef: epsilon*Ef**2/2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,Ef: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman60', 'DescriptiveName': 'Feynman60, Lecture II.10.9', 'Formula_Str': 'sigma_den/epsilon*1/(1+chi)', 'Formula': 'sigma_den/epsilon*1/(1+chi)', 'Formula_Lambda': 'lambda args : (lambda sigma_den,epsilon,chi: sigma_den/epsilon*1/(1+chi) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda sigma_den,epsilon,chi: {0} )(*args)', 'Variables': [{'name': 'sigma_den', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'chi', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman61', 'DescriptiveName': 'Feynman61, Lecture II.11.3', 'Formula_Str': 'q*Ef/(m*(omega_0**2-omega**2))', 'Formula': 'q*Ef/(m*(omega_0**2-omega**2))', 'Formula_Lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: q*Ef/(m*(omega_0**2-omega**2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,Ef,m,omega_0,omega: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'omega_0', 'low': 3.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Feynman62', 'DescriptiveName': 'Feynman62, Lecture II.11.17', 'Formula_Str': 'n_0*(1+p_d*Ef*cos(theta)/(kb*T))', 'Formula': 'n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))', 'Formula_Lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: n_0*(1+p_d*Ef*np.cos(theta)/(kb*T)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: {0} )(*args)', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman63', 'DescriptiveName': 'Feynman63, Lecture II.11.20', 'Formula_Str': 'n_rho*p_d**2*Ef/(3*kb*T)', 'Formula': 'n_rho*p_d**2*Ef/(3*kb*T)', 'Formula_Lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: n_rho*p_d**2*Ef/(3*kb*T) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: {0} )(*args)', 'Variables': [{'name': 'n_rho', 'low': 1.0, 'high': 5.0}, {'name': 'p_d', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman64', 'DescriptiveName': 'Feynman64, Lecture II.11.27', 'Formula_Str': 'n*alpha/(1-(n*alpha/3))*epsilon*Ef', 'Formula': 'n*alpha/(1-(n*alpha/3))*epsilon*Ef', 'Formula_Lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: n*alpha/(1-(n*alpha/3))*epsilon*Ef )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,alpha,epsilon,Ef: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'alpha', 'low': 0.0, 'high': 1.0}, {'name': 'epsilon', 'low': 1.0, 'high': 2.0}, {'name': 'Ef', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Feynman65', 'DescriptiveName': 'Feynman65, Lecture II.11.28', 'Formula_Str': '1+n*alpha/(1-(n*alpha/3))', 'Formula': '1+n*alpha/(1-(n*alpha/3))', 'Formula_Lambda': 'lambda args : (lambda n,alpha: 1+n*alpha/(1-(n*alpha/3)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,alpha: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'alpha', 'low': 0.0, 'high': 1.0}]},
{'EquationName': 'Feynman66', 'DescriptiveName': 'Feynman66, Lecture II.13.17', 'Formula_Str': '1/(4*pi*epsilon*c**2)*2*I/r', 'Formula': '1/(4*np.pi*epsilon*c**2)*2*I/r', 'Formula_Lambda': 'lambda args : (lambda epsilon,c,I,r: 1/(4*np.pi*epsilon*c**2)*2*I/r )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,c,I,r: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'I', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman67', 'DescriptiveName': 'Feynman67, Lecture II.13.23', 'Formula_Str': 'rho_c_0/sqrt(1-v**2/c**2)', 'Formula': 'rho_c_0/np.sqrt(1-v**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0/np.sqrt(1-v**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda rho_c_0,v,c: {0} )(*args)', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman68', 'DescriptiveName': 'Feynman68, Lecture II.13.34', 'Formula_Str': 'rho_c_0*v/sqrt(1-v**2/c**2)', 'Formula': 'rho_c_0*v/np.sqrt(1-v**2/c**2)', 'Formula_Lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0*v/np.sqrt(1-v**2/c**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda rho_c_0,v,c: {0} )(*args)', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman69', 'DescriptiveName': 'Feynman69, Lecture II.15.4', 'Formula_Str': '-mom*B*cos(theta)', 'Formula': '-mom*B*np.cos(theta)', 'Formula_Lambda': 'lambda args : (lambda mom,B,theta: -mom*B*np.cos(theta) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mom,B,theta: {0} )(*args)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman70', 'DescriptiveName': 'Feynman70, Lecture II.15.5', 'Formula_Str': '-p_d*Ef*cos(theta)', 'Formula': '-p_d*Ef*np.cos(theta)', 'Formula_Lambda': 'lambda args : (lambda p_d,Ef,theta: -p_d*Ef*np.cos(theta) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda p_d,Ef,theta: {0} )(*args)', 'Variables': [{'name': 'p_d', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman71', 'DescriptiveName': 'Feynman71, Lecture II.21.32', 'Formula_Str': 'q/(4*pi*epsilon*r*(1-v/c))', 'Formula': 'q/(4*np.pi*epsilon*r*(1-v/c))', 'Formula_Lambda': 'lambda args : (lambda q,epsilon,r,v,c: q/(4*np.pi*epsilon*r*(1-v/c)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,epsilon,r,v,c: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'EquationName': 'Feynman72', 'DescriptiveName': 'Feynman72, Lecture II.24.17', 'Formula_Str': 'sqrt(omega**2/c**2-pi**2/d**2)', 'Formula': 'np.sqrt(omega**2/c**2-np.pi**2/d**2)', 'Formula_Lambda': 'lambda args : (lambda omega,c,d: np.sqrt(omega**2/c**2-np.pi**2/d**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda omega,c,d: {0} )(*args)', 'Variables': [{'name': 'omega', 'low': 4.0, 'high': 6.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 2.0, 'high': 4.0}]},
{'EquationName': 'Feynman73', 'DescriptiveName': 'Feynman73, Lecture II.27.16', 'Formula_Str': 'epsilon*c*Ef**2', 'Formula': 'epsilon*c*Ef**2', 'Formula_Lambda': 'lambda args : (lambda epsilon,c,Ef: epsilon*c*Ef**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,c,Ef: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman74', 'DescriptiveName': 'Feynman74, Lecture II.27.18', 'Formula_Str': 'epsilon*Ef**2', 'Formula': 'epsilon*Ef**2', 'Formula_Lambda': 'lambda args : (lambda epsilon,Ef: epsilon*Ef**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,Ef: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman75', 'DescriptiveName': 'Feynman75, Lecture II.34.2a', 'Formula_Str': 'q*v/(2*pi*r)', 'Formula': 'q*v/(2*np.pi*r)', 'Formula_Lambda': 'lambda args : (lambda q,v,r: q*v/(2*np.pi*r) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,v,r: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman76', 'DescriptiveName': 'Feynman76, Lecture II.34.2', 'Formula_Str': 'q*v*r/2', 'Formula': 'q*v*r/2', 'Formula_Lambda': 'lambda args : (lambda q,v,r: q*v*r/2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,v,r: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman77', 'DescriptiveName': 'Feynman77, Lecture II.34.11', 'Formula_Str': 'g_*q*B/(2*m)', 'Formula': 'g_*q*B/(2*m)', 'Formula_Lambda': 'lambda args : (lambda g_,q,B,m: g_*q*B/(2*m) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda g_,q,B,m: {0} )(*args)', 'Variables': [{'name': 'g_', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman78', 'DescriptiveName': 'Feynman78, Lecture II.34.29a', 'Formula_Str': 'q*h/(4*pi*m)', 'Formula': 'q*h/(4*np.pi*m)', 'Formula_Lambda': 'lambda args : (lambda q,h,m: q*h/(4*np.pi*m) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,h,m: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman79', 'DescriptiveName': 'Feynman79, Lecture II.34.29b', 'Formula_Str': 'g_*mom*B*Jz/(h/(2*pi))', 'Formula': 'g_*mom*B*Jz/(h/(2*np.pi))', 'Formula_Lambda': 'lambda args : (lambda g_,h,Jz,mom,B: g_*mom*B*Jz/(h/(2*np.pi)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda g_,h,Jz,mom,B: {0} )(*args)', 'Variables': [{'name': 'g_', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'Jz', 'low': 1.0, 'high': 5.0}, {'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman80', 'DescriptiveName': 'Feynman80, Lecture II.35.18', 'Formula_Str': 'n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))', 'Formula': 'n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))', 'Formula_Lambda': 'lambda args : (lambda n_0,kb,T,mom,B: n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T))) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n_0,kb,T,mom,B: {0} )(*args)', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'mom', 'low': 1.0, 'high': 3.0}, {'name': 'B', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman81', 'DescriptiveName': 'Feynman81, Lecture II.35.21', 'Formula_Str': 'n_rho*mom*tanh(mom*B/(kb*T))', 'Formula': 'n_rho*mom*np.tanh(mom*B/(kb*T))', 'Formula_Lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: n_rho*mom*np.tanh(mom*B/(kb*T)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n_rho,mom,B,kb,T: {0} )(*args)', 'Variables': [{'name': 'n_rho', 'low': 1.0, 'high': 5.0}, {'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman82', 'DescriptiveName': 'Feynman82, Lecture II.36.38', 'Formula_Str': 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M', 'Formula': 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M', 'Formula_Lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: {0} )(*args)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 3.0}, {'name': 'H', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}, {'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'c', 'low': 1.0, 'high': 3.0}, {'name': 'M', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Feynman83', 'DescriptiveName': 'Feynman83, Lecture II.37.1', 'Formula_Str': 'mom*(1+chi)*B', 'Formula': 'mom*(1+chi)*B', 'Formula_Lambda': 'lambda args : (lambda mom,B,chi: mom*(1+chi)*B )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mom,B,chi: {0} )(*args)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'chi', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman84', 'DescriptiveName': 'Feynman84, Lecture II.38.3', 'Formula_Str': 'Y*A*x/d', 'Formula': 'Y*A*x/d', 'Formula_Lambda': 'lambda args : (lambda Y,A,d,x: Y*A*x/d )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Y,A,d,x: {0} )(*args)', 'Variables': [{'name': 'Y', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman85', 'DescriptiveName': 'Feynman85, Lecture II.38.14', 'Formula_Str': 'Y/(2*(1+sigma))', 'Formula': 'Y/(2*(1+sigma))', 'Formula_Lambda': 'lambda args : (lambda Y,sigma: Y/(2*(1+sigma)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Y,sigma: {0} )(*args)', 'Variables': [{'name': 'Y', 'low': 1.0, 'high': 5.0}, {'name': 'sigma', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman86', 'DescriptiveName': 'Feynman86, Lecture III.4.32', 'Formula_Str': '1/(exp((h/(2*pi))*omega/(kb*T))-1)', 'Formula': '1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)', 'Formula_Lambda': 'lambda args : (lambda h,omega,kb,T: 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda h,omega,kb,T: {0} )(*args)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman87', 'DescriptiveName': 'Feynman87, Lecture III.4.33', 'Formula_Str': '(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)', 'Formula': '(h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)', 'Formula_Lambda': 'lambda args : (lambda h,omega,kb,T: (h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda h,omega,kb,T: {0} )(*args)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman88', 'DescriptiveName': 'Feynman88, Lecture III.7.38', 'Formula_Str': '2*mom*B/(h/(2*pi))', 'Formula': '2*mom*B/(h/(2*np.pi))', 'Formula_Lambda': 'lambda args : (lambda mom,B,h: 2*mom*B/(h/(2*np.pi)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mom,B,h: {0} )(*args)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman89', 'DescriptiveName': 'Feynman89, Lecture III.8.54', 'Formula_Str': 'sin(E_n*t/(h/(2*pi)))**2', 'Formula': 'np.sin(E_n*t/(h/(2*np.pi)))**2', 'Formula_Lambda': 'lambda args : (lambda E_n,t,h: np.sin(E_n*t/(h/(2*np.pi)))**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda E_n,t,h: {0} )(*args)', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 2.0}, {'name': 't', 'low': 1.0, 'high': 2.0}, {'name': 'h', 'low': 1.0, 'high': 4.0}]},
{'EquationName': 'Feynman90', 'DescriptiveName': 'Feynman90, Lecture III.9.52', 'Formula_Str': '(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2', 'Formula': '(p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2', 'Formula_Lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: {0} )(*args)', 'Variables': [{'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}, {'name': 't', 'low': 1.0, 'high': 3.0}, {'name': 'h', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman91', 'DescriptiveName': 'Feynman91, Lecture III.10.19', 'Formula_Str': 'mom*sqrt(Bx**2+By**2+Bz**2)', 'Formula': 'mom*np.sqrt(Bx**2+By**2+Bz**2)', 'Formula_Lambda': 'lambda args : (lambda mom,Bx,By,Bz: mom*np.sqrt(Bx**2+By**2+Bz**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda mom,Bx,By,Bz: {0} )(*args)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'Bx', 'low': 1.0, 'high': 5.0}, {'name': 'By', 'low': 1.0, 'high': 5.0}, {'name': 'Bz', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman92', 'DescriptiveName': 'Feynman92, Lecture III.12.43', 'Formula_Str': 'n*(h/(2*pi))', 'Formula': 'n*(h/(2*np.pi))', 'Formula_Lambda': 'lambda args : (lambda n,h: n*(h/(2*np.pi)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda n,h: {0} )(*args)', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman93', 'DescriptiveName': 'Feynman93, Lecture III.13.18', 'Formula_Str': '2*E_n*d**2*k/(h/(2*pi))', 'Formula': '2*E_n*d**2*k/(h/(2*np.pi))', 'Formula_Lambda': 'lambda args : (lambda E_n,d,k,h: 2*E_n*d**2*k/(h/(2*np.pi)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda E_n,d,k,h: {0} )(*args)', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'k', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman94', 'DescriptiveName': 'Feynman94, Lecture III.14.14', 'Formula_Str': 'I_0*(exp(q*Volt/(kb*T))-1)', 'Formula': 'I_0*(np.exp(q*Volt/(kb*T))-1)', 'Formula_Lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*(np.exp(q*Volt/(kb*T))-1) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda I_0,q,Volt,kb,T: {0} )(*args)', 'Variables': [{'name': 'I_0', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 2.0}, {'name': 'Volt', 'low': 1.0, 'high': 2.0}, {'name': 'kb', 'low': 1.0, 'high': 2.0}, {'name': 'T', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Feynman95', 'DescriptiveName': 'Feynman95, Lecture III.15.12', 'Formula_Str': '2*U*(1-cos(k*d))', 'Formula': '2*U*(1-np.cos(k*d))', 'Formula_Lambda': 'lambda args : (lambda U,k,d: 2*U*(1-np.cos(k*d)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda U,k,d: {0} )(*args)', 'Variables': [{'name': 'U', 'low': 1.0, 'high': 5.0}, {'name': 'k', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman96', 'DescriptiveName': 'Feynman96, Lecture III.15.14', 'Formula_Str': '(h/(2*pi))**2/(2*E_n*d**2)', 'Formula': '(h/(2*np.pi))**2/(2*E_n*d**2)', 'Formula_Lambda': 'lambda args : (lambda h,E_n,d: (h/(2*np.pi))**2/(2*E_n*d**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda h,E_n,d: {0} )(*args)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'E_n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman97', 'DescriptiveName': 'Feynman97, Lecture III.15.27', 'Formula_Str': '2*pi*alpha/(n*d)', 'Formula': '2*np.pi*alpha/(n*d)', 'Formula_Lambda': 'lambda args : (lambda alpha,n,d: 2*np.pi*alpha/(n*d) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda alpha,n,d: {0} )(*args)', 'Variables': [{'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman98', 'DescriptiveName': 'Feynman98, Lecture III.17.37', 'Formula_Str': 'beta*(1+alpha*cos(theta))', 'Formula': 'beta*(1+alpha*np.cos(theta))', 'Formula_Lambda': 'lambda args : (lambda beta,alpha,theta: beta*(1+alpha*np.cos(theta)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda beta,alpha,theta: {0} )(*args)', 'Variables': [{'name': 'beta', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman99', 'DescriptiveName': 'Feynman99, Lecture III.19.51', 'Formula_Str': '-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)', 'Formula': '-m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)', 'Formula_Lambda': 'lambda args : (lambda m,q,h,n,epsilon: -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,q,h,n,epsilon: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Feynman100', 'DescriptiveName': 'Feynman100, Lecture III.21.20', 'Formula_Str': '-rho_c_0*q*A_vec/m', 'Formula': '-rho_c_0*q*A_vec/m', 'Formula_Lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: -rho_c_0*q*A_vec/m )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda rho_c_0,q,A_vec,m: {0} )(*args)', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'A_vec', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus1', 'DescriptiveName': 'Bonus1.0, Rutherford scattering', 'Formula_Str': '(Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2', 'Formula': '(Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2', 'Formula_Lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: (Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: {0} )(*args)', 'Variables': [{'name': 'Z_1', 'low': 1.0, 'high': 2.0}, {'name': 'Z_2', 'low': 1.0, 'high': 2.0}, {'name': 'alpha', 'low': 1.0, 'high': 2.0}, {'name': 'hbar', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus2', 'DescriptiveName': 'Bonus2.0, 3.55 Goldstein', 'Formula_Str': 'm*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))', 'Formula': 'm*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2))', 'Formula_Lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: m*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'k_G', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'theta1', 'low': 0.0, 'high': 6.0}, {'name': 'theta2', 'low': 0.0, 'high': 6.0}]},
{'EquationName': 'Bonus3', 'DescriptiveName': 'Bonus3.0, 3.64 Goldstein', 'Formula_Str': 'd*(1-alpha**2)/(1+alpha*cos(theta1-theta2))', 'Formula': 'd*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2))', 'Formula_Lambda': 'lambda args : (lambda d,alpha,theta1,theta2: d*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda d,alpha,theta1,theta2: {0} )(*args)', 'Variables': [{'name': 'd', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 2.0, 'high': 4.0}, {'name': 'theta1', 'low': 4.0, 'high': 5.0}, {'name': 'theta2', 'low': 4.0, 'high': 5.0}]},
{'EquationName': 'Bonus4', 'DescriptiveName': 'Bonus4.0, 3.16 Goldstein', 'Formula_Str': 'sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))', 'Formula': 'np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))', 'Formula_Lambda': 'lambda args : (lambda m,E_n,U,L,r: np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2))) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,E_n,U,L,r: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 8.0, 'high': 12.0}, {'name': 'U', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus5', 'DescriptiveName': 'Bonus5.0, 3.74 Goldstein', 'Formula_Str': '2*pi*d**(3/2)/sqrt(G*(m1+m2))', 'Formula': '2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2))', 'Formula_Lambda': 'lambda args : (lambda d,G,m1,m2: 2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda d,G,m1,m2: {0} )(*args)', 'Variables': [{'name': 'd', 'low': 1.0, 'high': 3.0}, {'name': 'G', 'low': 1.0, 'high': 3.0}, {'name': 'm1', 'low': 1.0, 'high': 3.0}, {'name': 'm2', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus6', 'DescriptiveName': 'Bonus6.0, 3.99 Goldstein', 'Formula_Str': 'sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))', 'Formula': 'np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))', 'Formula_Lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: {0} )(*args)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'Z_1', 'low': 1.0, 'high': 3.0}, {'name': 'Z_2', 'low': 1.0, 'high': 3.0}, {'name': 'q', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus7', 'DescriptiveName': 'Bonus7.0, Friedman Equation', 'Formula_Str': 'sqrt(8*pi*G*rho/3-alpha*c**2/d**2)', 'Formula': 'np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2)', 'Formula_Lambda': 'lambda args : (lambda G,rho,alpha,c,d: np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda G,rho,alpha,c,d: {0} )(*args)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 3.0}, {'name': 'rho', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus8', 'DescriptiveName': 'Bonus8.0, Compton Scattering', 'Formula_Str': 'E_n/(1+E_n/(m*c**2)*(1-cos(theta)))', 'Formula': 'E_n/(1+E_n/(m*c**2)*(1-np.cos(theta)))', 'Formula_Lambda': 'lambda args : (lambda E_n,m,c,theta: E_n/(1+E_n/(m*c**2)*(1-np.cos(theta))) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda E_n,m,c,theta: {0} )(*args)', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'c', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus9', 'DescriptiveName': 'Bonus9.0, Gravitational wave ratiated power', 'Formula_Str': '-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5', 'Formula': '-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5', 'Formula_Lambda': 'lambda args : (lambda G,c,m1,m2,r: -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda G,c,m1,m2,r: {0} )(*args)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Bonus10', 'DescriptiveName': 'Bonus10.0, Relativistic aberation', 'Formula_Str': 'arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))', 'Formula': 'np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2)))', 'Formula_Lambda': 'lambda args : (lambda c,v,theta2: np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2))) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda c,v,theta2: {0} )(*args)', 'Variables': [{'name': 'c', 'low': 4.0, 'high': 6.0}, {'name': 'v', 'low': 1.0, 'high': 3.0}, {'name': 'theta2', 'low': 1.0, 'high': 3.0}]},
{'EquationName': 'Bonus11', 'DescriptiveName': 'Bonus11.0, N-slit diffraction', 'Formula_Str': 'I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2', 'Formula': 'I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2', 'Formula_Lambda': 'lambda args : (lambda I_0,alpha,delta,n: I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2 )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda I_0,alpha,delta,n: {0} )(*args)', 'Variables': [{'name': 'I_0', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}, {'name': 'delta', 'low': 1.0, 'high': 3.0}, {'name': 'n', 'low': 1.0, 'high': 2.0}]},
{'EquationName': 'Bonus12', 'DescriptiveName': 'Bonus12.0, 2.11 Jackson', 'Formula_Str': 'q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)', 'Formula': 'q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)', 'Formula_Lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,y,Volt,d,epsilon: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'y', 'low': 1.0, 'high': 3.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 4.0, 'high': 6.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus13', 'DescriptiveName': 'Bonus13.0, 3.45 Jackson', 'Formula_Str': '1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))', 'Formula': '1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha))', 'Formula_Lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: 1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda q,r,d,alpha,epsilon: {0} )(*args)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}, {'name': 'd', 'low': 4.0, 'high': 6.0}, {'name': 'alpha', 'low': 0.0, 'high': 6.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus14', 'DescriptiveName': "Bonus14.0, 4.60' Jackson", 'Formula_Str': 'Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))', 'Formula': 'Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))', 'Formula_Lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda Ef,theta,r,d,alpha: {0} )(*args)', 'Variables': [{'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 6.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus15', 'DescriptiveName': 'Bonus15.0, 11.38 Jackson', 'Formula_Str': 'sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))', 'Formula': 'np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta))', 'Formula_Lambda': 'lambda args : (lambda c,v,omega,theta: np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda c,v,omega,theta: {0} )(*args)', 'Variables': [{'name': 'c', 'low': 5.0, 'high': 20.0}, {'name': 'v', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 6.0}]},
{'EquationName': 'Bonus16', 'DescriptiveName': 'Bonus16.0, 8.56 Goldstein', 'Formula_Str': 'sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt', 'Formula': 'np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt', 'Formula_Lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,c,p,q,A_vec,Volt: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'A_vec', 'low': 1.0, 'high': 5.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus17', 'DescriptiveName': "Bonus17.0, 12.80' Goldstein", 'Formula_Str': '1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))', 'Formula': '1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))', 'Formula_Lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda m,omega,p,y,x,alpha: {0} )(*args)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}, {'name': 'y', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus18', 'DescriptiveName': 'Bonus18.0, 15.2.1 Weinberg', 'Formula_Str': '3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)', 'Formula': '3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2)', 'Formula_Lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda G,k_f,r,H_G,c: {0} )(*args)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 5.0}, {'name': 'k_f', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'H_G', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus19', 'DescriptiveName': 'Bonus19.0, 15.2.2 Weinberg', 'Formula_Str': '-1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))', 'Formula': '-1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))', 'Formula_Lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: -1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha)) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: {0} )(*args)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 5.0}, {'name': 'k_f', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'H_G', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'EquationName': 'Bonus20', 'DescriptiveName': 'Bonus20.0, Klein-Nishina (13.132 Schwarz)', 'Formula_Str': '1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)', 'Formula': '1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2)', 'Formula_Lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2) )(*args)', 'Formula_Lambda_Stump': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: {0} )(*args)', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'beta', 'low': 0.0, 'high': 6.0}]} 
 ]