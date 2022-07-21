import pandas as pd
import numpy as np

def Noise(target, noise_level = None):
  if( (noise_level == None) | (noise_level == 0)):
    return target
  assert 0 < noise_level < 1, f"Argument '{noise_level=}' out of range"

  stdDev = np.std(target)
  noise = np.random.normal(0,stdDev*np.sqrt(noise_level/(1-noise_level)),len(target))
  return target + noise


def Feynman1_data(size = 10000, noise_level = 0):
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
    return Feynman1(theta,noise_level)

def Feynman1(theta, noise_level = 0):
    """
    Feynman1, Lecture I.6.2a

    Arguments:
        theta: float or array-like, default range (1.0,3.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: exp(-theta**2/2)/sqrt(2*pi)
    """
    target = np.exp(-theta**2/2)/np.sqrt(2*np.pi)
    return pd.DataFrame(
      list(
        zip(
          theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['theta','f']
    )
  

def Feynman2_data(size = 10000, noise_level = 0):
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
    return Feynman2(sigma,theta,noise_level)

def Feynman2(sigma,theta, noise_level = 0):
    """
    Feynman2, Lecture I.6.2

    Arguments:
        sigma: float or array-like, default range (1.0,3.0)
        theta: float or array-like, default range (1.0,3.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)
    """
    target = np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
    return pd.DataFrame(
      list(
        zip(
          sigma,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['sigma','theta','f']
    )
  

def Feynman3_data(size = 10000, noise_level = 0):
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
    return Feynman3(sigma,theta,theta1,noise_level)

def Feynman3(sigma,theta,theta1, noise_level = 0):
    """
    Feynman3, Lecture I.6.2b

    Arguments:
        sigma: float or array-like, default range (1.0,3.0)
        theta: float or array-like, default range (1.0,3.0)
        theta1: float or array-like, default range (1.0,3.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)
    """
    target = np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)
    return pd.DataFrame(
      list(
        zip(
          sigma,theta,theta1
          ,Noise(target,noise_level)
        )
      )
      ,columns=['sigma','theta','theta1','f']
    )
  

def Feynman4_data(size = 10000, noise_level = 0):
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
    return Feynman4(x1,x2,y1,y2,noise_level)

def Feynman4(x1,x2,y1,y2, noise_level = 0):
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
        f: sqrt((x2-x1)**2+(y2-y1)**2)
    """
    target = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return pd.DataFrame(
      list(
        zip(
          x1,x2,y1,y2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x1','x2','y1','y2','d']
    )
  

def Feynman5_data(size = 10000, noise_level = 0):
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
    return Feynman5(m1,m2,G,x1,x2,y1,y2,z1,z2,noise_level)

def Feynman5(m1,m2,G,x1,x2,y1,y2,z1,z2, noise_level = 0):
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
        f: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    """
    target = G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    return pd.DataFrame(
      list(
        zip(
          m1,m2,G,x1,x2,y1,y2,z1,z2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m1','m2','G','x1','x2','y1','y2','z1','z2','F']
    )
  

def Feynman6_data(size = 10000, noise_level = 0):
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
    return Feynman6(m_0,v,c,noise_level)

def Feynman6(m_0,v,c, noise_level = 0):
    """
    Feynman6, Lecture I.10.7

    Arguments:
        m_0: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,2.0)
        c: float or array-like, default range (3.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: m_0/sqrt(1-v**2/c**2)
    """
    target = m_0/np.sqrt(1-v**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          m_0,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m_0','v','c','m']
    )
  

def Feynman7_data(size = 10000, noise_level = 0):
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
    return Feynman7(x1,x2,x3,y1,y2,y3,noise_level)

def Feynman7(x1,x2,x3,y1,y2,y3, noise_level = 0):
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
        f: x1*y1+x2*y2+x3*y3
    """
    target = x1*y1+x2*y2+x3*y3
    return pd.DataFrame(
      list(
        zip(
          x1,x2,x3,y1,y2,y3
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x1','x2','x3','y1','y2','y3','A']
    )
  

def Feynman8_data(size = 10000, noise_level = 0):
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
    return Feynman8(mu,Nn,noise_level)

def Feynman8(mu,Nn, noise_level = 0):
    """
    Feynman8, Lecture I.12.1

    Arguments:
        mu: float or array-like, default range (1.0,5.0)
        Nn: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: mu*Nn
    """
    target = mu*Nn
    return pd.DataFrame(
      list(
        zip(
          mu,Nn
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mu','Nn','F']
    )
  

def Feynman10_data(size = 10000, noise_level = 0):
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
    return Feynman10(q1,q2,epsilon,r,noise_level)

def Feynman10(q1,q2,epsilon,r, noise_level = 0):
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
        f: q1*q2*r/(4*pi*epsilon*r**3)
    """
    target = q1*q2*r/(4*np.pi*epsilon*r**3)
    return pd.DataFrame(
      list(
        zip(
          q1,q2,epsilon,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q1','q2','epsilon','r','F']
    )
  

def Feynman11_data(size = 10000, noise_level = 0):
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
    return Feynman11(q1,epsilon,r,noise_level)

def Feynman11(q1,epsilon,r, noise_level = 0):
    """
    Feynman11, Lecture I.12.4

    Arguments:
        q1: float or array-like, default range (1.0,5.0)
        epsilon: float or array-like, default range (1.0,5.0)
        r: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q1*r/(4*pi*epsilon*r**3)
    """
    target = q1*r/(4*np.pi*epsilon*r**3)
    return pd.DataFrame(
      list(
        zip(
          q1,epsilon,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q1','epsilon','r','Ef']
    )
  

def Feynman12_data(size = 10000, noise_level = 0):
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
    return Feynman12(q2,Ef,noise_level)

def Feynman12(q2,Ef, noise_level = 0):
    """
    Feynman12, Lecture I.12.5

    Arguments:
        q2: float or array-like, default range (1.0,5.0)
        Ef: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q2*Ef
    """
    target = q2*Ef
    return pd.DataFrame(
      list(
        zip(
          q2,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q2','Ef','F']
    )
  

def Feynman13_data(size = 10000, noise_level = 0):
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
    return Feynman13(q,Ef,B,v,theta,noise_level)

def Feynman13(q,Ef,B,v,theta, noise_level = 0):
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
        f: q*(Ef+B*v*sin(theta))
    """
    target = q*(Ef+B*v*np.sin(theta))
    return pd.DataFrame(
      list(
        zip(
          q,Ef,B,v,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','Ef','B','v','theta','F']
    )
  

def Feynman9_data(size = 10000, noise_level = 0):
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
    return Feynman9(m,v,u,w,noise_level)

def Feynman9(m,v,u,w, noise_level = 0):
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
        f: 1/2*m*(v**2+u**2+w**2)
    """
    target = 1/2*m*(v**2+u**2+w**2)
    return pd.DataFrame(
      list(
        zip(
          m,v,u,w
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','v','u','w','K']
    )
  

def Feynman14_data(size = 10000, noise_level = 0):
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
    return Feynman14(m1,m2,r1,r2,G,noise_level)

def Feynman14(m1,m2,r1,r2,G, noise_level = 0):
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
        f: G*m1*m2*(1/r2-1/r1)
    """
    target = G*m1*m2*(1/r2-1/r1)
    return pd.DataFrame(
      list(
        zip(
          m1,m2,r1,r2,G
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m1','m2','r1','r2','G','U']
    )
  

def Feynman15_data(size = 10000, noise_level = 0):
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
    return Feynman15(m,g,z,noise_level)

def Feynman15(m,g,z, noise_level = 0):
    """
    Feynman15, Lecture I.14.3

    Arguments:
        m: float or array-like, default range (1.0,5.0)
        g: float or array-like, default range (1.0,5.0)
        z: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: m*g*z
    """
    target = m*g*z
    return pd.DataFrame(
      list(
        zip(
          m,g,z
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','g','z','U']
    )
  

def Feynman16_data(size = 10000, noise_level = 0):
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
    return Feynman16(k_spring,x,noise_level)

def Feynman16(k_spring,x, noise_level = 0):
    """
    Feynman16, Lecture I.14.4

    Arguments:
        k_spring: float or array-like, default range (1.0,5.0)
        x: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 1/2*k_spring*x**2
    """
    target = 1/2*k_spring*x**2
    return pd.DataFrame(
      list(
        zip(
          k_spring,x
          ,Noise(target,noise_level)
        )
      )
      ,columns=['k_spring','x','U']
    )
  

def Feynman17_data(size = 10000, noise_level = 0):
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
    return Feynman17(x,u,c,t,noise_level)

def Feynman17(x,u,c,t, noise_level = 0):
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
        f: (x-u*t)/sqrt(1-u**2/c**2)
    """
    target = (x-u*t)/np.sqrt(1-u**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          x,u,c,t
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x','u','c','t','x1']
    )
  

def Feynman18_data(size = 10000, noise_level = 0):
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
    return Feynman18(x,c,u,t,noise_level)

def Feynman18(x,c,u,t, noise_level = 0):
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
        f: (t-u*x/c**2)/sqrt(1-u**2/c**2)
    """
    target = (t-u*x/c**2)/np.sqrt(1-u**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          x,c,u,t
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x','c','u','t','t1']
    )
  

def Feynman19_data(size = 10000, noise_level = 0):
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
    return Feynman19(m_0,v,c,noise_level)

def Feynman19(m_0,v,c, noise_level = 0):
    """
    Feynman19, Lecture I.15.1

    Arguments:
        m_0: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,2.0)
        c: float or array-like, default range (3.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: m_0*v/sqrt(1-v**2/c**2)
    """
    target = m_0*v/np.sqrt(1-v**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          m_0,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m_0','v','c','p']
    )
  

def Feynman20_data(size = 10000, noise_level = 0):
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
    return Feynman20(c,v,u,noise_level)

def Feynman20(c,v,u, noise_level = 0):
    """
    Feynman20, Lecture I.16.6

    Arguments:
        c: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,5.0)
        u: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: (u+v)/(1+u*v/c**2)
    """
    target = (u+v)/(1+u*v/c**2)
    return pd.DataFrame(
      list(
        zip(
          c,v,u
          ,Noise(target,noise_level)
        )
      )
      ,columns=['c','v','u','v1']
    )
  

def Feynman21_data(size = 10000, noise_level = 0):
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
    return Feynman21(m1,m2,r1,r2,noise_level)

def Feynman21(m1,m2,r1,r2, noise_level = 0):
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
        f: (m1*r1+m2*r2)/(m1+m2)
    """
    target = (m1*r1+m2*r2)/(m1+m2)
    return pd.DataFrame(
      list(
        zip(
          m1,m2,r1,r2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m1','m2','r1','r2','r']
    )
  

def Feynman22_data(size = 10000, noise_level = 0):
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
    return Feynman22(r,F,theta,noise_level)

def Feynman22(r,F,theta, noise_level = 0):
    """
    Feynman22, Lecture I.18.12

    Arguments:
        r: float or array-like, default range (1.0,5.0)
        F: float or array-like, default range (1.0,5.0)
        theta: float or array-like, default range (0.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: r*F*sin(theta)
    """
    target = r*F*np.sin(theta)
    return pd.DataFrame(
      list(
        zip(
          r,F,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['r','F','theta','tau']
    )
  

def Feynman23_data(size = 10000, noise_level = 0):
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
    return Feynman23(m,r,v,theta,noise_level)

def Feynman23(m,r,v,theta, noise_level = 0):
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
        f: m*r*v*sin(theta)
    """
    target = m*r*v*np.sin(theta)
    return pd.DataFrame(
      list(
        zip(
          m,r,v,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','r','v','theta','L']
    )
  

def Feynman24_data(size = 10000, noise_level = 0):
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
    return Feynman24(m,omega,omega_0,x,noise_level)

def Feynman24(m,omega,omega_0,x, noise_level = 0):
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
        f: 1/2*m*(omega**2+omega_0**2)*1/2*x**2
    """
    target = 1/2*m*(omega**2+omega_0**2)*1/2*x**2
    return pd.DataFrame(
      list(
        zip(
          m,omega,omega_0,x
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','omega','omega_0','x','E_n']
    )
  

def Feynman25_data(size = 10000, noise_level = 0):
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
    return Feynman25(q,C,noise_level)

def Feynman25(q,C, noise_level = 0):
    """
    Feynman25, Lecture I.25.13

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        C: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q/C
    """
    target = q/C
    return pd.DataFrame(
      list(
        zip(
          q,C
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','C','Volt']
    )
  

def Feynman26_data(size = 10000, noise_level = 0):
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
    return Feynman26(n,theta2,noise_level)

def Feynman26(n,theta2, noise_level = 0):
    """
    Feynman26, Lecture I.26.2

    Arguments:
        n: float or array-like, default range (0.0,1.0)
        theta2: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: arcsin(n*sin(theta2))
    """
    target = np.arcsin(n*np.sin(theta2))
    return pd.DataFrame(
      list(
        zip(
          n,theta2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','theta2','theta1']
    )
  

def Feynman27_data(size = 10000, noise_level = 0):
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
    return Feynman27(d1,d2,n,noise_level)

def Feynman27(d1,d2,n, noise_level = 0):
    """
    Feynman27, Lecture I.27.6

    Arguments:
        d1: float or array-like, default range (1.0,5.0)
        d2: float or array-like, default range (1.0,5.0)
        n: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 1/(1/d1+n/d2)
    """
    target = 1/(1/d1+n/d2)
    return pd.DataFrame(
      list(
        zip(
          d1,d2,n
          ,Noise(target,noise_level)
        )
      )
      ,columns=['d1','d2','n','foc']
    )
  

def Feynman28_data(size = 10000, noise_level = 0):
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
    return Feynman28(omega,c,noise_level)

def Feynman28(omega,c, noise_level = 0):
    """
    Feynman28, Lecture I.29.4

    Arguments:
        omega: float or array-like, default range (1.0,10.0)
        c: float or array-like, default range (1.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: omega/c
    """
    target = omega/c
    return pd.DataFrame(
      list(
        zip(
          omega,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['omega','c','k']
    )
  

def Feynman29_data(size = 10000, noise_level = 0):
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
    return Feynman29(x1,x2,theta1,theta2,noise_level)

def Feynman29(x1,x2,theta1,theta2, noise_level = 0):
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
        f: sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
    """
    target = np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))
    return pd.DataFrame(
      list(
        zip(
          x1,x2,theta1,theta2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x1','x2','theta1','theta2','x']
    )
  

def Feynman30_data(size = 10000, noise_level = 0):
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
    return Feynman30(Int_0,theta,n,noise_level)

def Feynman30(Int_0,theta,n, noise_level = 0):
    """
    Feynman30, Lecture I.30.3

    Arguments:
        Int_0: float or array-like, default range (1.0,5.0)
        theta: float or array-like, default range (1.0,5.0)
        n: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: Int_0*sin(n*theta/2)**2/sin(theta/2)**2
    """
    target = Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2
    return pd.DataFrame(
      list(
        zip(
          Int_0,theta,n
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Int_0','theta','n','Int']
    )
  

def Feynman31_data(size = 10000, noise_level = 0):
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
    return Feynman31(lambd,d,n,noise_level)

def Feynman31(lambd,d,n, noise_level = 0):
    """
    Feynman31, Lecture I.30.5

    Arguments:
        lambd: float or array-like, default range (1.0,2.0)
        d: float or array-like, default range (2.0,5.0)
        n: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: arcsin(lambd/(n*d))
    """
    target = np.arcsin(lambd/(n*d))
    return pd.DataFrame(
      list(
        zip(
          lambd,d,n
          ,Noise(target,noise_level)
        )
      )
      ,columns=['lambd','d','n','theta']
    )
  

def Feynman32_data(size = 10000, noise_level = 0):
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
    return Feynman32(q,a,epsilon,c,noise_level)

def Feynman32(q,a,epsilon,c, noise_level = 0):
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
        f: q**2*a**2/(6*pi*epsilon*c**3)
    """
    target = q**2*a**2/(6*np.pi*epsilon*c**3)
    return pd.DataFrame(
      list(
        zip(
          q,a,epsilon,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','a','epsilon','c','Pwr']
    )
  

def Feynman33_data(size = 10000, noise_level = 0):
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
    return Feynman33(epsilon,c,Ef,r,omega,omega_0,noise_level)

def Feynman33(epsilon,c,Ef,r,omega,omega_0, noise_level = 0):
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
        f: (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
    """
    target = (1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)
    return pd.DataFrame(
      list(
        zip(
          epsilon,c,Ef,r,omega,omega_0
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','c','Ef','r','omega','omega_0','Pwr']
    )
  

def Feynman34_data(size = 10000, noise_level = 0):
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
    return Feynman34(q,v,B,p,noise_level)

def Feynman34(q,v,B,p, noise_level = 0):
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
        f: q*v*B/p
    """
    target = q*v*B/p
    return pd.DataFrame(
      list(
        zip(
          q,v,B,p
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','v','B','p','omega']
    )
  

def Feynman35_data(size = 10000, noise_level = 0):
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
    return Feynman35(c,v,omega_0,noise_level)

def Feynman35(c,v,omega_0, noise_level = 0):
    """
    Feynman35, Lecture I.34.1

    Arguments:
        c: float or array-like, default range (3.0,10.0)
        v: float or array-like, default range (1.0,2.0)
        omega_0: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: omega_0/(1-v/c)
    """
    target = omega_0/(1-v/c)
    return pd.DataFrame(
      list(
        zip(
          c,v,omega_0
          ,Noise(target,noise_level)
        )
      )
      ,columns=['c','v','omega_0','omega']
    )
  

def Feynman36_data(size = 10000, noise_level = 0):
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
    return Feynman36(c,v,omega_0,noise_level)

def Feynman36(c,v,omega_0, noise_level = 0):
    """
    Feynman36, Lecture I.34.14

    Arguments:
        c: float or array-like, default range (3.0,10.0)
        v: float or array-like, default range (1.0,2.0)
        omega_0: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: (1+v/c)/sqrt(1-v**2/c**2)*omega_0
    """
    target = (1+v/c)/np.sqrt(1-v**2/c**2)*omega_0
    return pd.DataFrame(
      list(
        zip(
          c,v,omega_0
          ,Noise(target,noise_level)
        )
      )
      ,columns=['c','v','omega_0','omega']
    )
  

def Feynman37_data(size = 10000, noise_level = 0):
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
    return Feynman37(omega,h,noise_level)

def Feynman37(omega,h, noise_level = 0):
    """
    Feynman37, Lecture I.34.27

    Arguments:
        omega: float or array-like, default range (1.0,5.0)
        h: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: (h/(2*pi))*omega
    """
    target = (h/(2*np.pi))*omega
    return pd.DataFrame(
      list(
        zip(
          omega,h
          ,Noise(target,noise_level)
        )
      )
      ,columns=['omega','h','E_n']
    )
  

def Feynman38_data(size = 10000, noise_level = 0):
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
    return Feynman38(I1,I2,delta,noise_level)

def Feynman38(I1,I2,delta, noise_level = 0):
    """
    Feynman38, Lecture I.37.4

    Arguments:
        I1: float or array-like, default range (1.0,5.0)
        I2: float or array-like, default range (1.0,5.0)
        delta: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: I1+I2+2*sqrt(I1*I2)*cos(delta)
    """
    target = I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)
    return pd.DataFrame(
      list(
        zip(
          I1,I2,delta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['I1','I2','delta','Int']
    )
  

def Feynman39_data(size = 10000, noise_level = 0):
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
    return Feynman39(m,q,h,epsilon,noise_level)

def Feynman39(m,q,h,epsilon, noise_level = 0):
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
        f: 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)
    """
    target = 4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)
    return pd.DataFrame(
      list(
        zip(
          m,q,h,epsilon
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','q','h','epsilon','r']
    )
  

def Feynman40_data(size = 10000, noise_level = 0):
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
    return Feynman40(pr,V,noise_level)

def Feynman40(pr,V, noise_level = 0):
    """
    Feynman40, Lecture I.39.1

    Arguments:
        pr: float or array-like, default range (1.0,5.0)
        V: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 3/2*pr*V
    """
    target = 3/2*pr*V
    return pd.DataFrame(
      list(
        zip(
          pr,V
          ,Noise(target,noise_level)
        )
      )
      ,columns=['pr','V','E_n']
    )
  

def Feynman41_data(size = 10000, noise_level = 0):
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
    return Feynman41(gamma,pr,V,noise_level)

def Feynman41(gamma,pr,V, noise_level = 0):
    """
    Feynman41, Lecture I.39.11

    Arguments:
        gamma: float or array-like, default range (2.0,5.0)
        pr: float or array-like, default range (1.0,5.0)
        V: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 1/(gamma-1)*pr*V
    """
    target = 1/(gamma-1)*pr*V
    return pd.DataFrame(
      list(
        zip(
          gamma,pr,V
          ,Noise(target,noise_level)
        )
      )
      ,columns=['gamma','pr','V','E_n']
    )
  

def Feynman42_data(size = 10000, noise_level = 0):
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
    return Feynman42(n,T,V,kb,noise_level)

def Feynman42(n,T,V,kb, noise_level = 0):
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
        f: n*kb*T/V
    """
    target = n*kb*T/V
    return pd.DataFrame(
      list(
        zip(
          n,T,V,kb
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','T','V','kb','pr']
    )
  

def Feynman43_data(size = 10000, noise_level = 0):
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
    return Feynman43(n_0,m,x,T,g,kb,noise_level)

def Feynman43(n_0,m,x,T,g,kb, noise_level = 0):
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
        f: n_0*exp(-m*g*x/(kb*T))
    """
    target = n_0*np.exp(-m*g*x/(kb*T))
    return pd.DataFrame(
      list(
        zip(
          n_0,m,x,T,g,kb
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n_0','m','x','T','g','kb','n']
    )
  

def Feynman44_data(size = 10000, noise_level = 0):
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
    return Feynman44(omega,T,h,kb,c,noise_level)

def Feynman44(omega,T,h,kb,c, noise_level = 0):
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
        f: h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))
    """
    target = h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))
    return pd.DataFrame(
      list(
        zip(
          omega,T,h,kb,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['omega','T','h','kb','c','L_rad']
    )
  

def Feynman45_data(size = 10000, noise_level = 0):
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
    return Feynman45(mu_drift,q,Volt,d,noise_level)

def Feynman45(mu_drift,q,Volt,d, noise_level = 0):
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
        f: mu_drift*q*Volt/d
    """
    target = mu_drift*q*Volt/d
    return pd.DataFrame(
      list(
        zip(
          mu_drift,q,Volt,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mu_drift','q','Volt','d','v']
    )
  

def Feynman46_data(size = 10000, noise_level = 0):
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
    return Feynman46(mob,T,kb,noise_level)

def Feynman46(mob,T,kb, noise_level = 0):
    """
    Feynman46, Lecture I.43.31

    Arguments:
        mob: float or array-like, default range (1.0,5.0)
        T: float or array-like, default range (1.0,5.0)
        kb: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: mob*kb*T
    """
    target = mob*kb*T
    return pd.DataFrame(
      list(
        zip(
          mob,T,kb
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mob','T','kb','D']
    )
  

def Feynman47_data(size = 10000, noise_level = 0):
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
    return Feynman47(gamma,kb,A,v,noise_level)

def Feynman47(gamma,kb,A,v, noise_level = 0):
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
        f: 1/(gamma-1)*kb*v/A
    """
    target = 1/(gamma-1)*kb*v/A
    return pd.DataFrame(
      list(
        zip(
          gamma,kb,A,v
          ,Noise(target,noise_level)
        )
      )
      ,columns=['gamma','kb','A','v','kappa']
    )
  

def Feynman48_data(size = 10000, noise_level = 0):
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
    return Feynman48(n,kb,T,V1,V2,noise_level)

def Feynman48(n,kb,T,V1,V2, noise_level = 0):
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
        f: n*kb*T*ln(V2/V1)
    """
    target = n*kb*T*np.ln(V2/V1)
    return pd.DataFrame(
      list(
        zip(
          n,kb,T,V1,V2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','kb','T','V1','V2','E_n']
    )
  

def Feynman49_data(size = 10000, noise_level = 0):
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
    return Feynman49(gamma,pr,rho,noise_level)

def Feynman49(gamma,pr,rho, noise_level = 0):
    """
    Feynman49, Lecture I.47.23

    Arguments:
        gamma: float or array-like, default range (1.0,5.0)
        pr: float or array-like, default range (1.0,5.0)
        rho: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: sqrt(gamma*pr/rho)
    """
    target = np.sqrt(gamma*pr/rho)
    return pd.DataFrame(
      list(
        zip(
          gamma,pr,rho
          ,Noise(target,noise_level)
        )
      )
      ,columns=['gamma','pr','rho','c']
    )
  

def Feynman50_data(size = 10000, noise_level = 0):
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
    return Feynman50(m,v,c,noise_level)

def Feynman50(m,v,c, noise_level = 0):
    """
    Feynman50, Lecture I.48.2

    Arguments:
        m: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,2.0)
        c: float or array-like, default range (3.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: m*c**2/sqrt(1-v**2/c**2)
    """
    target = m*c**2/np.sqrt(1-v**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          m,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','v','c','E_n']
    )
  

def Feynman51_data(size = 10000, noise_level = 0):
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
    return Feynman51(x1,omega,t,alpha,noise_level)

def Feynman51(x1,omega,t,alpha, noise_level = 0):
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
        f: x1*(cos(omega*t)+alpha*cos(omega*t)**2)
    """
    target = x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)
    return pd.DataFrame(
      list(
        zip(
          x1,omega,t,alpha
          ,Noise(target,noise_level)
        )
      )
      ,columns=['x1','omega','t','alpha','x']
    )
  

def Feynman52_data(size = 10000, noise_level = 0):
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
    return Feynman52(kappa,T1,T2,A,d,noise_level)

def Feynman52(kappa,T1,T2,A,d, noise_level = 0):
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
        f: kappa*(T2-T1)*A/d
    """
    target = kappa*(T2-T1)*A/d
    return pd.DataFrame(
      list(
        zip(
          kappa,T1,T2,A,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['kappa','T1','T2','A','d','Pwr']
    )
  

def Feynman53_data(size = 10000, noise_level = 0):
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
    return Feynman53(Pwr,r,noise_level)

def Feynman53(Pwr,r, noise_level = 0):
    """
    Feynman53, Lecture II.3.24

    Arguments:
        Pwr: float or array-like, default range (1.0,5.0)
        r: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: Pwr/(4*pi*r**2)
    """
    target = Pwr/(4*np.pi*r**2)
    return pd.DataFrame(
      list(
        zip(
          Pwr,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Pwr','r','flux']
    )
  

def Feynman54_data(size = 10000, noise_level = 0):
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
    return Feynman54(q,epsilon,r,noise_level)

def Feynman54(q,epsilon,r, noise_level = 0):
    """
    Feynman54, Lecture II.4.23

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        epsilon: float or array-like, default range (1.0,5.0)
        r: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q/(4*pi*epsilon*r)
    """
    target = q/(4*np.pi*epsilon*r)
    return pd.DataFrame(
      list(
        zip(
          q,epsilon,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','epsilon','r','Volt']
    )
  

def Feynman55_data(size = 10000, noise_level = 0):
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
    return Feynman55(epsilon,p_d,theta,r,noise_level)

def Feynman55(epsilon,p_d,theta,r, noise_level = 0):
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
        f: 1/(4*pi*epsilon)*p_d*cos(theta)/r**2
    """
    target = 1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2
    return pd.DataFrame(
      list(
        zip(
          epsilon,p_d,theta,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','p_d','theta','r','Volt']
    )
  

def Feynman56_data(size = 10000, noise_level = 0):
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
    return Feynman56(epsilon,p_d,r,x,y,z,noise_level)

def Feynman56(epsilon,p_d,r,x,y,z, noise_level = 0):
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
        f: p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)
    """
    target = p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)
    return pd.DataFrame(
      list(
        zip(
          epsilon,p_d,r,x,y,z
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','p_d','r','x','y','z','Ef']
    )
  

def Feynman57_data(size = 10000, noise_level = 0):
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
    return Feynman57(epsilon,p_d,theta,r,noise_level)

def Feynman57(epsilon,p_d,theta,r, noise_level = 0):
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
        f: p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3
    """
    target = p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3
    return pd.DataFrame(
      list(
        zip(
          epsilon,p_d,theta,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','p_d','theta','r','Ef']
    )
  

def Feynman58_data(size = 10000, noise_level = 0):
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
    return Feynman58(q,epsilon,d,noise_level)

def Feynman58(q,epsilon,d, noise_level = 0):
    """
    Feynman58, Lecture II.8.7

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        epsilon: float or array-like, default range (1.0,5.0)
        d: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 3/5*q**2/(4*pi*epsilon*d)
    """
    target = 3/5*q**2/(4*np.pi*epsilon*d)
    return pd.DataFrame(
      list(
        zip(
          q,epsilon,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','epsilon','d','E_n']
    )
  

def Feynman59_data(size = 10000, noise_level = 0):
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
    return Feynman59(epsilon,Ef,noise_level)

def Feynman59(epsilon,Ef, noise_level = 0):
    """
    Feynman59, Lecture II.8.31

    Arguments:
        epsilon: float or array-like, default range (1.0,5.0)
        Ef: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: epsilon*Ef**2/2
    """
    target = epsilon*Ef**2/2
    return pd.DataFrame(
      list(
        zip(
          epsilon,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','Ef','E_den']
    )
  

def Feynman60_data(size = 10000, noise_level = 0):
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
    return Feynman60(sigma_den,epsilon,chi,noise_level)

def Feynman60(sigma_den,epsilon,chi, noise_level = 0):
    """
    Feynman60, Lecture II.10.9

    Arguments:
        sigma_den: float or array-like, default range (1.0,5.0)
        epsilon: float or array-like, default range (1.0,5.0)
        chi: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: sigma_den/epsilon*1/(1+chi)
    """
    target = sigma_den/epsilon*1/(1+chi)
    return pd.DataFrame(
      list(
        zip(
          sigma_den,epsilon,chi
          ,Noise(target,noise_level)
        )
      )
      ,columns=['sigma_den','epsilon','chi','Ef']
    )
  

def Feynman61_data(size = 10000, noise_level = 0):
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
    return Feynman61(q,Ef,m,omega_0,omega,noise_level)

def Feynman61(q,Ef,m,omega_0,omega, noise_level = 0):
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
        f: q*Ef/(m*(omega_0**2-omega**2))
    """
    target = q*Ef/(m*(omega_0**2-omega**2))
    return pd.DataFrame(
      list(
        zip(
          q,Ef,m,omega_0,omega
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','Ef','m','omega_0','omega','x']
    )
  

def Feynman62_data(size = 10000, noise_level = 0):
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
    return Feynman62(n_0,kb,T,theta,p_d,Ef,noise_level)

def Feynman62(n_0,kb,T,theta,p_d,Ef, noise_level = 0):
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
        f: n_0*(1+p_d*Ef*cos(theta)/(kb*T))
    """
    target = n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))
    return pd.DataFrame(
      list(
        zip(
          n_0,kb,T,theta,p_d,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n_0','kb','T','theta','p_d','Ef','n']
    )
  

def Feynman63_data(size = 10000, noise_level = 0):
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
    return Feynman63(n_rho,p_d,Ef,kb,T,noise_level)

def Feynman63(n_rho,p_d,Ef,kb,T, noise_level = 0):
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
        f: n_rho*p_d**2*Ef/(3*kb*T)
    """
    target = n_rho*p_d**2*Ef/(3*kb*T)
    return pd.DataFrame(
      list(
        zip(
          n_rho,p_d,Ef,kb,T
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n_rho','p_d','Ef','kb','T','Pol']
    )
  

def Feynman64_data(size = 10000, noise_level = 0):
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
    return Feynman64(n,alpha,epsilon,Ef,noise_level)

def Feynman64(n,alpha,epsilon,Ef, noise_level = 0):
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
        f: n*alpha/(1-(n*alpha/3))*epsilon*Ef
    """
    target = n*alpha/(1-(n*alpha/3))*epsilon*Ef
    return pd.DataFrame(
      list(
        zip(
          n,alpha,epsilon,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','alpha','epsilon','Ef','Pol']
    )
  

def Feynman65_data(size = 10000, noise_level = 0):
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
    return Feynman65(n,alpha,noise_level)

def Feynman65(n,alpha, noise_level = 0):
    """
    Feynman65, Lecture II.11.28

    Arguments:
        n: float or array-like, default range (0.0,1.0)
        alpha: float or array-like, default range (0.0,1.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 1+n*alpha/(1-(n*alpha/3))
    """
    target = 1+n*alpha/(1-(n*alpha/3))
    return pd.DataFrame(
      list(
        zip(
          n,alpha
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','alpha','theta']
    )
  

def Feynman66_data(size = 10000, noise_level = 0):
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
    return Feynman66(epsilon,c,I,r,noise_level)

def Feynman66(epsilon,c,I,r, noise_level = 0):
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
        f: 1/(4*pi*epsilon*c**2)*2*I/r
    """
    target = 1/(4*np.pi*epsilon*c**2)*2*I/r
    return pd.DataFrame(
      list(
        zip(
          epsilon,c,I,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','c','I','r','B']
    )
  

def Feynman67_data(size = 10000, noise_level = 0):
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
    return Feynman67(rho_c_0,v,c,noise_level)

def Feynman67(rho_c_0,v,c, noise_level = 0):
    """
    Feynman67, Lecture II.13.23

    Arguments:
        rho_c_0: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,2.0)
        c: float or array-like, default range (3.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: rho_c_0/sqrt(1-v**2/c**2)
    """
    target = rho_c_0/np.sqrt(1-v**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          rho_c_0,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['rho_c_0','v','c','rho_c']
    )
  

def Feynman68_data(size = 10000, noise_level = 0):
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
    return Feynman68(rho_c_0,v,c,noise_level)

def Feynman68(rho_c_0,v,c, noise_level = 0):
    """
    Feynman68, Lecture II.13.34

    Arguments:
        rho_c_0: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,2.0)
        c: float or array-like, default range (3.0,10.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: rho_c_0*v/sqrt(1-v**2/c**2)
    """
    target = rho_c_0*v/np.sqrt(1-v**2/c**2)
    return pd.DataFrame(
      list(
        zip(
          rho_c_0,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['rho_c_0','v','c','j']
    )
  

def Feynman69_data(size = 10000, noise_level = 0):
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
    return Feynman69(mom,B,theta,noise_level)

def Feynman69(mom,B,theta, noise_level = 0):
    """
    Feynman69, Lecture II.15.4

    Arguments:
        mom: float or array-like, default range (1.0,5.0)
        B: float or array-like, default range (1.0,5.0)
        theta: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: -mom*B*cos(theta)
    """
    target = -mom*B*np.cos(theta)
    return pd.DataFrame(
      list(
        zip(
          mom,B,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mom','B','theta','E_n']
    )
  

def Feynman70_data(size = 10000, noise_level = 0):
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
    return Feynman70(p_d,Ef,theta,noise_level)

def Feynman70(p_d,Ef,theta, noise_level = 0):
    """
    Feynman70, Lecture II.15.5

    Arguments:
        p_d: float or array-like, default range (1.0,5.0)
        Ef: float or array-like, default range (1.0,5.0)
        theta: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: -p_d*Ef*cos(theta)
    """
    target = -p_d*Ef*np.cos(theta)
    return pd.DataFrame(
      list(
        zip(
          p_d,Ef,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['p_d','Ef','theta','E_n']
    )
  

def Feynman71_data(size = 10000, noise_level = 0):
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
    return Feynman71(q,epsilon,r,v,c,noise_level)

def Feynman71(q,epsilon,r,v,c, noise_level = 0):
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
        f: q/(4*pi*epsilon*r*(1-v/c))
    """
    target = q/(4*np.pi*epsilon*r*(1-v/c))
    return pd.DataFrame(
      list(
        zip(
          q,epsilon,r,v,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','epsilon','r','v','c','Volt']
    )
  

def Feynman72_data(size = 10000, noise_level = 0):
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
    return Feynman72(omega,c,d,noise_level)

def Feynman72(omega,c,d, noise_level = 0):
    """
    Feynman72, Lecture II.24.17

    Arguments:
        omega: float or array-like, default range (4.0,6.0)
        c: float or array-like, default range (1.0,2.0)
        d: float or array-like, default range (2.0,4.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: sqrt(omega**2/c**2-pi**2/d**2)
    """
    target = np.sqrt(omega**2/c**2-np.pi**2/d**2)
    return pd.DataFrame(
      list(
        zip(
          omega,c,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['omega','c','d','k']
    )
  

def Feynman73_data(size = 10000, noise_level = 0):
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
    return Feynman73(epsilon,c,Ef,noise_level)

def Feynman73(epsilon,c,Ef, noise_level = 0):
    """
    Feynman73, Lecture II.27.16

    Arguments:
        epsilon: float or array-like, default range (1.0,5.0)
        c: float or array-like, default range (1.0,5.0)
        Ef: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: epsilon*c*Ef**2
    """
    target = epsilon*c*Ef**2
    return pd.DataFrame(
      list(
        zip(
          epsilon,c,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','c','Ef','flux']
    )
  

def Feynman74_data(size = 10000, noise_level = 0):
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
    return Feynman74(epsilon,Ef,noise_level)

def Feynman74(epsilon,Ef, noise_level = 0):
    """
    Feynman74, Lecture II.27.18

    Arguments:
        epsilon: float or array-like, default range (1.0,5.0)
        Ef: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: epsilon*Ef**2
    """
    target = epsilon*Ef**2
    return pd.DataFrame(
      list(
        zip(
          epsilon,Ef
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','Ef','E_den']
    )
  

def Feynman75_data(size = 10000, noise_level = 0):
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
    return Feynman75(q,v,r,noise_level)

def Feynman75(q,v,r, noise_level = 0):
    """
    Feynman75, Lecture II.34.2a

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,5.0)
        r: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q*v/(2*pi*r)
    """
    target = q*v/(2*np.pi*r)
    return pd.DataFrame(
      list(
        zip(
          q,v,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','v','r','I']
    )
  

def Feynman76_data(size = 10000, noise_level = 0):
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
    return Feynman76(q,v,r,noise_level)

def Feynman76(q,v,r, noise_level = 0):
    """
    Feynman76, Lecture II.34.2

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        v: float or array-like, default range (1.0,5.0)
        r: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q*v*r/2
    """
    target = q*v*r/2
    return pd.DataFrame(
      list(
        zip(
          q,v,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','v','r','mom']
    )
  

def Feynman77_data(size = 10000, noise_level = 0):
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
    return Feynman77(g_,q,B,m,noise_level)

def Feynman77(g_,q,B,m, noise_level = 0):
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
        f: g_*q*B/(2*m)
    """
    target = g_*q*B/(2*m)
    return pd.DataFrame(
      list(
        zip(
          g_,q,B,m
          ,Noise(target,noise_level)
        )
      )
      ,columns=['g_','q','B','m','omega']
    )
  

def Feynman78_data(size = 10000, noise_level = 0):
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
    return Feynman78(q,h,m,noise_level)

def Feynman78(q,h,m, noise_level = 0):
    """
    Feynman78, Lecture II.34.29a

    Arguments:
        q: float or array-like, default range (1.0,5.0)
        h: float or array-like, default range (1.0,5.0)
        m: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: q*h/(4*pi*m)
    """
    target = q*h/(4*np.pi*m)
    return pd.DataFrame(
      list(
        zip(
          q,h,m
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','h','m','mom']
    )
  

def Feynman79_data(size = 10000, noise_level = 0):
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
    return Feynman79(g_,h,Jz,mom,B,noise_level)

def Feynman79(g_,h,Jz,mom,B, noise_level = 0):
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
        f: g_*mom*B*Jz/(h/(2*pi))
    """
    target = g_*mom*B*Jz/(h/(2*np.pi))
    return pd.DataFrame(
      list(
        zip(
          g_,h,Jz,mom,B
          ,Noise(target,noise_level)
        )
      )
      ,columns=['g_','h','Jz','mom','B','E_n']
    )
  

def Feynman80_data(size = 10000, noise_level = 0):
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
    return Feynman80(n_0,kb,T,mom,B,noise_level)

def Feynman80(n_0,kb,T,mom,B, noise_level = 0):
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
        f: n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))
    """
    target = n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))
    return pd.DataFrame(
      list(
        zip(
          n_0,kb,T,mom,B
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n_0','kb','T','mom','B','n']
    )
  

def Feynman81_data(size = 10000, noise_level = 0):
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
    return Feynman81(n_rho,mom,B,kb,T,noise_level)

def Feynman81(n_rho,mom,B,kb,T, noise_level = 0):
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
        f: n_rho*mom*tanh(mom*B/(kb*T))
    """
    target = n_rho*mom*np.tanh(mom*B/(kb*T))
    return pd.DataFrame(
      list(
        zip(
          n_rho,mom,B,kb,T
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n_rho','mom','B','kb','T','M']
    )
  

def Feynman82_data(size = 10000, noise_level = 0):
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
    return Feynman82(mom,H,kb,T,alpha,epsilon,c,M,noise_level)

def Feynman82(mom,H,kb,T,alpha,epsilon,c,M, noise_level = 0):
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
        f: mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
    """
    target = mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M
    return pd.DataFrame(
      list(
        zip(
          mom,H,kb,T,alpha,epsilon,c,M
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mom','H','kb','T','alpha','epsilon','c','M','f']
    )
  

def Feynman83_data(size = 10000, noise_level = 0):
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
    return Feynman83(mom,B,chi,noise_level)

def Feynman83(mom,B,chi, noise_level = 0):
    """
    Feynman83, Lecture II.37.1

    Arguments:
        mom: float or array-like, default range (1.0,5.0)
        B: float or array-like, default range (1.0,5.0)
        chi: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: mom*(1+chi)*B
    """
    target = mom*(1+chi)*B
    return pd.DataFrame(
      list(
        zip(
          mom,B,chi
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mom','B','chi','E_n']
    )
  

def Feynman84_data(size = 10000, noise_level = 0):
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
    return Feynman84(Y,A,d,x,noise_level)

def Feynman84(Y,A,d,x, noise_level = 0):
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
        f: Y*A*x/d
    """
    target = Y*A*x/d
    return pd.DataFrame(
      list(
        zip(
          Y,A,d,x
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Y','A','d','x','F']
    )
  

def Feynman85_data(size = 10000, noise_level = 0):
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
    return Feynman85(Y,sigma,noise_level)

def Feynman85(Y,sigma, noise_level = 0):
    """
    Feynman85, Lecture II.38.14

    Arguments:
        Y: float or array-like, default range (1.0,5.0)
        sigma: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: Y/(2*(1+sigma))
    """
    target = Y/(2*(1+sigma))
    return pd.DataFrame(
      list(
        zip(
          Y,sigma
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Y','sigma','mu_S']
    )
  

def Feynman86_data(size = 10000, noise_level = 0):
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
    return Feynman86(h,omega,kb,T,noise_level)

def Feynman86(h,omega,kb,T, noise_level = 0):
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
        f: 1/(exp((h/(2*pi))*omega/(kb*T))-1)
    """
    target = 1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
    return pd.DataFrame(
      list(
        zip(
          h,omega,kb,T
          ,Noise(target,noise_level)
        )
      )
      ,columns=['h','omega','kb','T','n']
    )
  

def Feynman87_data(size = 10000, noise_level = 0):
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
    return Feynman87(h,omega,kb,T,noise_level)

def Feynman87(h,omega,kb,T, noise_level = 0):
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
        f: (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)
    """
    target = (h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)
    return pd.DataFrame(
      list(
        zip(
          h,omega,kb,T
          ,Noise(target,noise_level)
        )
      )
      ,columns=['h','omega','kb','T','E_n']
    )
  

def Feynman88_data(size = 10000, noise_level = 0):
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
    return Feynman88(mom,B,h,noise_level)

def Feynman88(mom,B,h, noise_level = 0):
    """
    Feynman88, Lecture III.7.38

    Arguments:
        mom: float or array-like, default range (1.0,5.0)
        B: float or array-like, default range (1.0,5.0)
        h: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 2*mom*B/(h/(2*pi))
    """
    target = 2*mom*B/(h/(2*np.pi))
    return pd.DataFrame(
      list(
        zip(
          mom,B,h
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mom','B','h','omega']
    )
  

def Feynman89_data(size = 10000, noise_level = 0):
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
    return Feynman89(E_n,t,h,noise_level)

def Feynman89(E_n,t,h, noise_level = 0):
    """
    Feynman89, Lecture III.8.54

    Arguments:
        E_n: float or array-like, default range (1.0,2.0)
        t: float or array-like, default range (1.0,2.0)
        h: float or array-like, default range (1.0,4.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: sin(E_n*t/(h/(2*pi)))**2
    """
    target = np.sin(E_n*t/(h/(2*np.pi)))**2
    return pd.DataFrame(
      list(
        zip(
          E_n,t,h
          ,Noise(target,noise_level)
        )
      )
      ,columns=['E_n','t','h','prob']
    )
  

def Feynman90_data(size = 10000, noise_level = 0):
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
    return Feynman90(p_d,Ef,t,h,omega,omega_0,noise_level)

def Feynman90(p_d,Ef,t,h,omega,omega_0, noise_level = 0):
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
        f: (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
    """
    target = (p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2
    return pd.DataFrame(
      list(
        zip(
          p_d,Ef,t,h,omega,omega_0
          ,Noise(target,noise_level)
        )
      )
      ,columns=['p_d','Ef','t','h','omega','omega_0','prob']
    )
  

def Feynman91_data(size = 10000, noise_level = 0):
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
    return Feynman91(mom,Bx,By,Bz,noise_level)

def Feynman91(mom,Bx,By,Bz, noise_level = 0):
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
        f: mom*sqrt(Bx**2+By**2+Bz**2)
    """
    target = mom*np.sqrt(Bx**2+By**2+Bz**2)
    return pd.DataFrame(
      list(
        zip(
          mom,Bx,By,Bz
          ,Noise(target,noise_level)
        )
      )
      ,columns=['mom','Bx','By','Bz','E_n']
    )
  

def Feynman92_data(size = 10000, noise_level = 0):
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
    return Feynman92(n,h,noise_level)

def Feynman92(n,h, noise_level = 0):
    """
    Feynman92, Lecture III.12.43

    Arguments:
        n: float or array-like, default range (1.0,5.0)
        h: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: n*(h/(2*pi))
    """
    target = n*(h/(2*np.pi))
    return pd.DataFrame(
      list(
        zip(
          n,h
          ,Noise(target,noise_level)
        )
      )
      ,columns=['n','h','L']
    )
  

def Feynman93_data(size = 10000, noise_level = 0):
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
    return Feynman93(E_n,d,k,h,noise_level)

def Feynman93(E_n,d,k,h, noise_level = 0):
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
        f: 2*E_n*d**2*k/(h/(2*pi))
    """
    target = 2*E_n*d**2*k/(h/(2*np.pi))
    return pd.DataFrame(
      list(
        zip(
          E_n,d,k,h
          ,Noise(target,noise_level)
        )
      )
      ,columns=['E_n','d','k','h','v']
    )
  

def Feynman94_data(size = 10000, noise_level = 0):
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
    return Feynman94(I_0,q,Volt,kb,T,noise_level)

def Feynman94(I_0,q,Volt,kb,T, noise_level = 0):
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
        f: I_0*(exp(q*Volt/(kb*T))-1)
    """
    target = I_0*(np.exp(q*Volt/(kb*T))-1)
    return pd.DataFrame(
      list(
        zip(
          I_0,q,Volt,kb,T
          ,Noise(target,noise_level)
        )
      )
      ,columns=['I_0','q','Volt','kb','T','I']
    )
  

def Feynman95_data(size = 10000, noise_level = 0):
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
    return Feynman95(U,k,d,noise_level)

def Feynman95(U,k,d, noise_level = 0):
    """
    Feynman95, Lecture III.15.12

    Arguments:
        U: float or array-like, default range (1.0,5.0)
        k: float or array-like, default range (1.0,5.0)
        d: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 2*U*(1-cos(k*d))
    """
    target = 2*U*(1-np.cos(k*d))
    return pd.DataFrame(
      list(
        zip(
          U,k,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['U','k','d','E_n']
    )
  

def Feynman96_data(size = 10000, noise_level = 0):
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
    return Feynman96(h,E_n,d,noise_level)

def Feynman96(h,E_n,d, noise_level = 0):
    """
    Feynman96, Lecture III.15.14

    Arguments:
        h: float or array-like, default range (1.0,5.0)
        E_n: float or array-like, default range (1.0,5.0)
        d: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: (h/(2*pi))**2/(2*E_n*d**2)
    """
    target = (h/(2*np.pi))**2/(2*E_n*d**2)
    return pd.DataFrame(
      list(
        zip(
          h,E_n,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['h','E_n','d','m']
    )
  

def Feynman97_data(size = 10000, noise_level = 0):
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
    return Feynman97(alpha,n,d,noise_level)

def Feynman97(alpha,n,d, noise_level = 0):
    """
    Feynman97, Lecture III.15.27

    Arguments:
        alpha: float or array-like, default range (1.0,5.0)
        n: float or array-like, default range (1.0,5.0)
        d: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: 2*pi*alpha/(n*d)
    """
    target = 2*np.pi*alpha/(n*d)
    return pd.DataFrame(
      list(
        zip(
          alpha,n,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['alpha','n','d','k']
    )
  

def Feynman98_data(size = 10000, noise_level = 0):
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
    return Feynman98(beta,alpha,theta,noise_level)

def Feynman98(beta,alpha,theta, noise_level = 0):
    """
    Feynman98, Lecture III.17.37

    Arguments:
        beta: float or array-like, default range (1.0,5.0)
        alpha: float or array-like, default range (1.0,5.0)
        theta: float or array-like, default range (1.0,5.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: beta*(1+alpha*cos(theta))
    """
    target = beta*(1+alpha*np.cos(theta))
    return pd.DataFrame(
      list(
        zip(
          beta,alpha,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['beta','alpha','theta','f']
    )
  

def Feynman99_data(size = 10000, noise_level = 0):
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
    return Feynman99(m,q,h,n,epsilon,noise_level)

def Feynman99(m,q,h,n,epsilon, noise_level = 0):
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
        f: -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)
    """
    target = -m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)
    return pd.DataFrame(
      list(
        zip(
          m,q,h,n,epsilon
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','q','h','n','epsilon','E_n']
    )
  

def Feynman100_data(size = 10000, noise_level = 0):
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
    return Feynman100(rho_c_0,q,A_vec,m,noise_level)

def Feynman100(rho_c_0,q,A_vec,m, noise_level = 0):
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
        f: -rho_c_0*q*A_vec/m
    """
    target = -rho_c_0*q*A_vec/m
    return pd.DataFrame(
      list(
        zip(
          rho_c_0,q,A_vec,m
          ,Noise(target,noise_level)
        )
      )
      ,columns=['rho_c_0','q','A_vec','m','j']
    )
  

def Bonus1_data(size = 10000, noise_level = 0):
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
    return Bonus1(Z_1,Z_2,alpha,hbar,c,E_n,theta,noise_level)

def Bonus1(Z_1,Z_2,alpha,hbar,c,E_n,theta, noise_level = 0):
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
        f: (Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2
    """
    target = (Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2
    return pd.DataFrame(
      list(
        zip(
          Z_1,Z_2,alpha,hbar,c,E_n,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Z_1','Z_2','alpha','hbar','c','E_n','theta','A']
    )
  

def Bonus2_data(size = 10000, noise_level = 0):
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
    return Bonus2(m,k_G,L,E_n,theta1,theta2,noise_level)

def Bonus2(m,k_G,L,E_n,theta1,theta2, noise_level = 0):
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
        f: m*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))
    """
    target = m*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2))
    return pd.DataFrame(
      list(
        zip(
          m,k_G,L,E_n,theta1,theta2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','k_G','L','E_n','theta1','theta2','k']
    )
  

def Bonus3_data(size = 10000, noise_level = 0):
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
    return Bonus3(d,alpha,theta1,theta2,noise_level)

def Bonus3(d,alpha,theta1,theta2, noise_level = 0):
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
        f: d*(1-alpha**2)/(1+alpha*cos(theta1-theta2))
    """
    target = d*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2))
    return pd.DataFrame(
      list(
        zip(
          d,alpha,theta1,theta2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['d','alpha','theta1','theta2','r']
    )
  

def Bonus4_data(size = 10000, noise_level = 0):
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
    return Bonus4(m,E_n,U,L,r,noise_level)

def Bonus4(m,E_n,U,L,r, noise_level = 0):
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
        f: sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))
    """
    target = np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))
    return pd.DataFrame(
      list(
        zip(
          m,E_n,U,L,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','E_n','U','L','r','v']
    )
  

def Bonus5_data(size = 10000, noise_level = 0):
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
    return Bonus5(d,G,m1,m2,noise_level)

def Bonus5(d,G,m1,m2, noise_level = 0):
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
        f: 2*pi*d**(3/2)/sqrt(G*(m1+m2))
    """
    target = 2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2))
    return pd.DataFrame(
      list(
        zip(
          d,G,m1,m2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['d','G','m1','m2','t']
    )
  

def Bonus6_data(size = 10000, noise_level = 0):
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
    return Bonus6(epsilon,L,m,Z_1,Z_2,q,E_n,noise_level)

def Bonus6(epsilon,L,m,Z_1,Z_2,q,E_n, noise_level = 0):
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
        f: sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))
    """
    target = np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))
    return pd.DataFrame(
      list(
        zip(
          epsilon,L,m,Z_1,Z_2,q,E_n
          ,Noise(target,noise_level)
        )
      )
      ,columns=['epsilon','L','m','Z_1','Z_2','q','E_n','alpha']
    )
  

def Bonus7_data(size = 10000, noise_level = 0):
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
    return Bonus7(G,rho,alpha,c,d,noise_level)

def Bonus7(G,rho,alpha,c,d, noise_level = 0):
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
        f: sqrt(8*pi*G*rho/3-alpha*c**2/d**2)
    """
    target = np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2)
    return pd.DataFrame(
      list(
        zip(
          G,rho,alpha,c,d
          ,Noise(target,noise_level)
        )
      )
      ,columns=['G','rho','alpha','c','d','H_G']
    )
  

def Bonus8_data(size = 10000, noise_level = 0):
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
    return Bonus8(E_n,m,c,theta,noise_level)

def Bonus8(E_n,m,c,theta, noise_level = 0):
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
        f: E_n/(1+E_n/(m*c**2)*(1-cos(theta)))
    """
    target = E_n/(1+E_n/(m*c**2)*(1-np.cos(theta)))
    return pd.DataFrame(
      list(
        zip(
          E_n,m,c,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['E_n','m','c','theta','K']
    )
  

def Bonus9_data(size = 10000, noise_level = 0):
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
    return Bonus9(G,c,m1,m2,r,noise_level)

def Bonus9(G,c,m1,m2,r, noise_level = 0):
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
        f: -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
    """
    target = -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
    return pd.DataFrame(
      list(
        zip(
          G,c,m1,m2,r
          ,Noise(target,noise_level)
        )
      )
      ,columns=['G','c','m1','m2','r','Pwr']
    )
  

def Bonus10_data(size = 10000, noise_level = 0):
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
    return Bonus10(c,v,theta2,noise_level)

def Bonus10(c,v,theta2, noise_level = 0):
    """
    Bonus10.0, Relativistic aberation

    Arguments:
        c: float or array-like, default range (4.0,6.0)
        v: float or array-like, default range (1.0,3.0)
        theta2: float or array-like, default range (1.0,3.0)
        noise_level: normal distributed noise added as target's 
              standard deviation times sqrt(noise_level/(1-noise_level))
    Returns:
        f: arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))
    """
    target = np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2)))
    return pd.DataFrame(
      list(
        zip(
          c,v,theta2
          ,Noise(target,noise_level)
        )
      )
      ,columns=['c','v','theta2','theta1']
    )
  

def Bonus11_data(size = 10000, noise_level = 0):
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
    return Bonus11(I_0,alpha,delta,n,noise_level)

def Bonus11(I_0,alpha,delta,n, noise_level = 0):
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
        f: I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2
    """
    target = I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2
    return pd.DataFrame(
      list(
        zip(
          I_0,alpha,delta,n
          ,Noise(target,noise_level)
        )
      )
      ,columns=['I_0','alpha','delta','n','I']
    )
  

def Bonus12_data(size = 10000, noise_level = 0):
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
    return Bonus12(q,y,Volt,d,epsilon,noise_level)

def Bonus12(q,y,Volt,d,epsilon, noise_level = 0):
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
        f: q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)
    """
    target = q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)
    return pd.DataFrame(
      list(
        zip(
          q,y,Volt,d,epsilon
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','y','Volt','d','epsilon','F']
    )
  

def Bonus13_data(size = 10000, noise_level = 0):
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
    return Bonus13(q,r,d,alpha,epsilon,noise_level)

def Bonus13(q,r,d,alpha,epsilon, noise_level = 0):
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
        f: 1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))
    """
    target = 1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha))
    return pd.DataFrame(
      list(
        zip(
          q,r,d,alpha,epsilon
          ,Noise(target,noise_level)
        )
      )
      ,columns=['q','r','d','alpha','epsilon','Volt']
    )
  

def Bonus14_data(size = 10000, noise_level = 0):
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
    return Bonus14(Ef,theta,r,d,alpha,noise_level)

def Bonus14(Ef,theta,r,d,alpha, noise_level = 0):
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
        f: Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))
    """
    target = Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))
    return pd.DataFrame(
      list(
        zip(
          Ef,theta,r,d,alpha
          ,Noise(target,noise_level)
        )
      )
      ,columns=['Ef','theta','r','d','alpha','Volt']
    )
  

def Bonus15_data(size = 10000, noise_level = 0):
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
    return Bonus15(c,v,omega,theta,noise_level)

def Bonus15(c,v,omega,theta, noise_level = 0):
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
        f: sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))
    """
    target = np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta))
    return pd.DataFrame(
      list(
        zip(
          c,v,omega,theta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['c','v','omega','theta','omega_0']
    )
  

def Bonus16_data(size = 10000, noise_level = 0):
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
    return Bonus16(m,c,p,q,A_vec,Volt,noise_level)

def Bonus16(m,c,p,q,A_vec,Volt, noise_level = 0):
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
        f: sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt
    """
    target = np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt
    return pd.DataFrame(
      list(
        zip(
          m,c,p,q,A_vec,Volt
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','c','p','q','A_vec','Volt','E_n']
    )
  

def Bonus17_data(size = 10000, noise_level = 0):
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
    return Bonus17(m,omega,p,y,x,alpha,noise_level)

def Bonus17(m,omega,p,y,x,alpha, noise_level = 0):
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
        f: 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))
    """
    target = 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))
    return pd.DataFrame(
      list(
        zip(
          m,omega,p,y,x,alpha
          ,Noise(target,noise_level)
        )
      )
      ,columns=['m','omega','p','y','x','alpha','E_n']
    )
  

def Bonus18_data(size = 10000, noise_level = 0):
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
    return Bonus18(G,k_f,r,H_G,c,noise_level)

def Bonus18(G,k_f,r,H_G,c, noise_level = 0):
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
        f: 3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)
    """
    target = 3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2)
    return pd.DataFrame(
      list(
        zip(
          G,k_f,r,H_G,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['G','k_f','r','H_G','c','rho_0']
    )
  

def Bonus19_data(size = 10000, noise_level = 0):
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
    return Bonus19(G,k_f,r,H_G,alpha,c,noise_level)

def Bonus19(G,k_f,r,H_G,alpha,c, noise_level = 0):
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
        f: -1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))
    """
    target = -1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))
    return pd.DataFrame(
      list(
        zip(
          G,k_f,r,H_G,alpha,c
          ,Noise(target,noise_level)
        )
      )
      ,columns=['G','k_f','r','H_G','alpha','c','pr']
    )
  

def Bonus20_data(size = 10000, noise_level = 0):
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
    return Bonus20(omega,omega_0,alpha,h,m,c,beta,noise_level)

def Bonus20(omega,omega_0,alpha,h,m,c,beta, noise_level = 0):
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
        f: 1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)
    """
    target = 1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2)
    return pd.DataFrame(
      list(
        zip(
          omega,omega_0,alpha,h,m,c,beta
          ,Noise(target,noise_level)
        )
      )
      ,columns=['omega','omega_0','alpha','h','m','c','beta','A']
    )
  
Functions = [
{'FunctionName': 'Feynman1', 'DescriptiveName': 'Feynman1, Lecture I.6.2a', 'Formula_Str': 'exp(-theta**2/2)/sqrt(2*pi)', 'Formula': 'np.exp(-theta**2/2)/np.sqrt(2*np.pi)', 'Variables': [{'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman2', 'DescriptiveName': 'Feynman2, Lecture I.6.2', 'Formula_Str': 'exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)', 'Formula': 'np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)', 'Variables': [{'name': 'sigma', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman3', 'DescriptiveName': 'Feynman3, Lecture I.6.2b', 'Formula_Str': 'exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)', 'Formula': 'np.exp(-((theta-theta1)/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)', 'Variables': [{'name': 'sigma', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'theta1', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman4', 'DescriptiveName': 'Feynman4, Lecture I.8.14', 'Formula_Str': 'sqrt((x2-x1)**2+(y2-y1)**2)', 'Formula': 'np.sqrt((x2-x1)**2+(y2-y1)**2)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'y1', 'low': 1.0, 'high': 5.0}, {'name': 'y2', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman5', 'DescriptiveName': 'Feynman5, Lecture I.9.18', 'Formula_Str': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)', 'Formula': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 2.0}, {'name': 'm2', 'low': 1.0, 'high': 2.0}, {'name': 'G', 'low': 1.0, 'high': 2.0}, {'name': 'x1', 'low': 3.0, 'high': 4.0}, {'name': 'x2', 'low': 1.0, 'high': 2.0}, {'name': 'y1', 'low': 3.0, 'high': 4.0}, {'name': 'y2', 'low': 1.0, 'high': 2.0}, {'name': 'z1', 'low': 3.0, 'high': 4.0}, {'name': 'z2', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Feynman6', 'DescriptiveName': 'Feynman6, Lecture I.10.7', 'Formula_Str': 'm_0/sqrt(1-v**2/c**2)', 'Formula': 'm_0/np.sqrt(1-v**2/c**2)', 'Variables': [{'name': 'm_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman7', 'DescriptiveName': 'Feynman7, Lecture I.11.19', 'Formula_Str': 'x1*y1+x2*y2+x3*y3', 'Formula': 'x1*y1+x2*y2+x3*y3', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'x3', 'low': 1.0, 'high': 5.0}, {'name': 'y1', 'low': 1.0, 'high': 5.0}, {'name': 'y2', 'low': 1.0, 'high': 5.0}, {'name': 'y3', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman8', 'DescriptiveName': 'Feynman8, Lecture I.12.1', 'Formula_Str': 'mu*Nn', 'Formula': 'mu*Nn', 'Variables': [{'name': 'mu', 'low': 1.0, 'high': 5.0}, {'name': 'Nn', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman10', 'DescriptiveName': 'Feynman10, Lecture I.12.2', 'Formula_Str': 'q1*q2*r/(4*pi*epsilon*r**3)', 'Formula': 'q1*q2*r/(4*np.pi*epsilon*r**3)', 'Variables': [{'name': 'q1', 'low': 1.0, 'high': 5.0}, {'name': 'q2', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman11', 'DescriptiveName': 'Feynman11, Lecture I.12.4', 'Formula_Str': 'q1*r/(4*pi*epsilon*r**3)', 'Formula': 'q1*r/(4*np.pi*epsilon*r**3)', 'Variables': [{'name': 'q1', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman12', 'DescriptiveName': 'Feynman12, Lecture I.12.5', 'Formula_Str': 'q2*Ef', 'Formula': 'q2*Ef', 'Variables': [{'name': 'q2', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman13', 'DescriptiveName': 'Feynman13, Lecture I.12.11', 'Formula_Str': 'q*(Ef+B*v*sin(theta))', 'Formula': 'q*(Ef+B*v*np.sin(theta))', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman9', 'DescriptiveName': 'Feynman9, Lecture I.13.4', 'Formula_Str': '1/2*m*(v**2+u**2+w**2)', 'Formula': '1/2*m*(v**2+u**2+w**2)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'u', 'low': 1.0, 'high': 5.0}, {'name': 'w', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman14', 'DescriptiveName': 'Feynman14, Lecture I.13.12', 'Formula_Str': 'G*m1*m2*(1/r2-1/r1)', 'Formula': 'G*m1*m2*(1/r2-1/r1)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r1', 'low': 1.0, 'high': 5.0}, {'name': 'r2', 'low': 1.0, 'high': 5.0}, {'name': 'G', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman15', 'DescriptiveName': 'Feynman15, Lecture I.14.3', 'Formula_Str': 'm*g*z', 'Formula': 'm*g*z', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'g', 'low': 1.0, 'high': 5.0}, {'name': 'z', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman16', 'DescriptiveName': 'Feynman16, Lecture I.14.4', 'Formula_Str': '1/2*k_spring*x**2', 'Formula': '1/2*k_spring*x**2', 'Variables': [{'name': 'k_spring', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman17', 'DescriptiveName': 'Feynman17, Lecture I.15.3x', 'Formula_Str': '(x-u*t)/sqrt(1-u**2/c**2)', 'Formula': '(x-u*t)/np.sqrt(1-u**2/c**2)', 'Variables': [{'name': 'x', 'low': 5.0, 'high': 10.0}, {'name': 'u', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 20.0}, {'name': 't', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Feynman18', 'DescriptiveName': 'Feynman18, Lecture I.15.3t', 'Formula_Str': '(t-u*x/c**2)/sqrt(1-u**2/c**2)', 'Formula': '(t-u*x/c**2)/np.sqrt(1-u**2/c**2)', 'Variables': [{'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'u', 'low': 1.0, 'high': 2.0}, {'name': 't', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman19', 'DescriptiveName': 'Feynman19, Lecture I.15.1', 'Formula_Str': 'm_0*v/sqrt(1-v**2/c**2)', 'Formula': 'm_0*v/np.sqrt(1-v**2/c**2)', 'Variables': [{'name': 'm_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman20', 'DescriptiveName': 'Feynman20, Lecture I.16.6', 'Formula_Str': '(u+v)/(1+u*v/c**2)', 'Formula': '(u+v)/(1+u*v/c**2)', 'Variables': [{'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'u', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman21', 'DescriptiveName': 'Feynman21, Lecture I.18.4', 'Formula_Str': '(m1*r1+m2*r2)/(m1+m2)', 'Formula': '(m1*r1+m2*r2)/(m1+m2)', 'Variables': [{'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r1', 'low': 1.0, 'high': 5.0}, {'name': 'r2', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman22', 'DescriptiveName': 'Feynman22, Lecture I.18.12', 'Formula_Str': 'r*F*sin(theta)', 'Formula': 'r*F*np.sin(theta)', 'Variables': [{'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'F', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 5.0}]},
{'FunctionName': 'Feynman23', 'DescriptiveName': 'Feynman23, Lecture I.18.14', 'Formula_Str': 'm*r*v*sin(theta)', 'Formula': 'm*r*v*np.sin(theta)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman24', 'DescriptiveName': 'Feynman24, Lecture I.24.6', 'Formula_Str': '1/2*m*(omega**2+omega_0**2)*1/2*x**2', 'Formula': '1/2*m*(omega**2+omega_0**2)*1/2*x**2', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 3.0}, {'name': 'omega_0', 'low': 1.0, 'high': 3.0}, {'name': 'x', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman25', 'DescriptiveName': 'Feynman25, Lecture I.25.13', 'Formula_Str': 'q/C', 'Formula': 'q/C', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'C', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman26', 'DescriptiveName': 'Feynman26, Lecture I.26.2', 'Formula_Str': 'arcsin(n*sin(theta2))', 'Formula': 'np.arcsin(n*np.sin(theta2))', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'theta2', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman27', 'DescriptiveName': 'Feynman27, Lecture I.27.6', 'Formula_Str': '1/(1/d1+n/d2)', 'Formula': '1/(1/d1+n/d2)', 'Variables': [{'name': 'd1', 'low': 1.0, 'high': 5.0}, {'name': 'd2', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman28', 'DescriptiveName': 'Feynman28, Lecture I.29.4', 'Formula_Str': 'omega/c', 'Formula': 'omega/c', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 10.0}, {'name': 'c', 'low': 1.0, 'high': 10.0}]},
{'FunctionName': 'Feynman29', 'DescriptiveName': 'Feynman29, Lecture I.29.16', 'Formula_Str': 'sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))', 'Formula': 'np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(theta1-theta2))', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 5.0}, {'name': 'x2', 'low': 1.0, 'high': 5.0}, {'name': 'theta1', 'low': 1.0, 'high': 5.0}, {'name': 'theta2', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman30', 'DescriptiveName': 'Feynman30, Lecture I.30.3', 'Formula_Str': 'Int_0*sin(n*theta/2)**2/sin(theta/2)**2', 'Formula': 'Int_0*np.sin(n*theta/2)**2/np.sin(theta/2)**2', 'Variables': [{'name': 'Int_0', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman31', 'DescriptiveName': 'Feynman31, Lecture I.30.5', 'Formula_Str': 'arcsin(lambd/(n*d))', 'Formula': 'np.arcsin(lambd/(n*d))', 'Variables': [{'name': 'lambd', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 2.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman32', 'DescriptiveName': 'Feynman32, Lecture I.32.5', 'Formula_Str': 'q**2*a**2/(6*pi*epsilon*c**3)', 'Formula': 'q**2*a**2/(6*np.pi*epsilon*c**3)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'a', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman33', 'DescriptiveName': 'Feynman33, Lecture I.32.17', 'Formula_Str': '(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)', 'Formula': '(1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'Ef', 'low': 1.0, 'high': 2.0}, {'name': 'r', 'low': 1.0, 'high': 2.0}, {'name': 'omega', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 3.0, 'high': 5.0}]},
{'FunctionName': 'Feynman34', 'DescriptiveName': 'Feynman34, Lecture I.34.8', 'Formula_Str': 'q*v*B/p', 'Formula': 'q*v*B/p', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman35', 'DescriptiveName': 'Feynman35, Lecture I.34.1', 'Formula_Str': 'omega_0/(1-v/c)', 'Formula': 'omega_0/(1-v/c)', 'Variables': [{'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman36', 'DescriptiveName': 'Feynman36, Lecture I.34.14', 'Formula_Str': '(1+v/c)/sqrt(1-v**2/c**2)*omega_0', 'Formula': '(1+v/c)/np.sqrt(1-v**2/c**2)*omega_0', 'Variables': [{'name': 'c', 'low': 3.0, 'high': 10.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman37', 'DescriptiveName': 'Feynman37, Lecture I.34.27', 'Formula_Str': '(h/(2*pi))*omega', 'Formula': '(h/(2*np.pi))*omega', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman38', 'DescriptiveName': 'Feynman38, Lecture I.37.4', 'Formula_Str': 'I1+I2+2*sqrt(I1*I2)*cos(delta)', 'Formula': 'I1+I2+2*np.sqrt(I1*I2)*np.cos(delta)', 'Variables': [{'name': 'I1', 'low': 1.0, 'high': 5.0}, {'name': 'I2', 'low': 1.0, 'high': 5.0}, {'name': 'delta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman39', 'DescriptiveName': 'Feynman39, Lecture I.38.12', 'Formula_Str': '4*pi*epsilon*(h/(2*pi))**2/(m*q**2)', 'Formula': '4*np.pi*epsilon*(h/(2*np.pi))**2/(m*q**2)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman40', 'DescriptiveName': 'Feynman40, Lecture I.39.1', 'Formula_Str': '3/2*pr*V', 'Formula': '3/2*pr*V', 'Variables': [{'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman41', 'DescriptiveName': 'Feynman41, Lecture I.39.11', 'Formula_Str': '1/(gamma-1)*pr*V', 'Formula': '1/(gamma-1)*pr*V', 'Variables': [{'name': 'gamma', 'low': 2.0, 'high': 5.0}, {'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman42', 'DescriptiveName': 'Feynman42, Lecture I.39.22', 'Formula_Str': 'n*kb*T/V', 'Formula': 'n*kb*T/V', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'V', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman43', 'DescriptiveName': 'Feynman43, Lecture I.40.1', 'Formula_Str': 'n_0*exp(-m*g*x/(kb*T))', 'Formula': 'n_0*np.exp(-m*g*x/(kb*T))', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'g', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman44', 'DescriptiveName': 'Feynman44, Lecture I.41.16', 'Formula_Str': 'h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))', 'Formula': 'h/(2*np.pi)*omega**3/(np.pi**2*c**2*(np.exp((h/(2*np.pi))*omega/(kb*T))-1))', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman45', 'DescriptiveName': 'Feynman45, Lecture I.43.16', 'Formula_Str': 'mu_drift*q*Volt/d', 'Formula': 'mu_drift*q*Volt/d', 'Variables': [{'name': 'mu_drift', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman46', 'DescriptiveName': 'Feynman46, Lecture I.43.31', 'Formula_Str': 'mob*kb*T', 'Formula': 'mob*kb*T', 'Variables': [{'name': 'mob', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman47', 'DescriptiveName': 'Feynman47, Lecture I.43.43', 'Formula_Str': '1/(gamma-1)*kb*v/A', 'Formula': '1/(gamma-1)*kb*v/A', 'Variables': [{'name': 'gamma', 'low': 2.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman48', 'DescriptiveName': 'Feynman48, Lecture I.44.4', 'Formula_Str': 'n*kb*T*ln(V2/V1)', 'Formula': 'n*kb*T*np.ln(V2/V1)', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}, {'name': 'V1', 'low': 1.0, 'high': 5.0}, {'name': 'V2', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman49', 'DescriptiveName': 'Feynman49, Lecture I.47.23', 'Formula_Str': 'sqrt(gamma*pr/rho)', 'Formula': 'np.sqrt(gamma*pr/rho)', 'Variables': [{'name': 'gamma', 'low': 1.0, 'high': 5.0}, {'name': 'pr', 'low': 1.0, 'high': 5.0}, {'name': 'rho', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman50', 'DescriptiveName': 'Feynman50, Lecture I.48.2', 'Formula_Str': 'm*c**2/sqrt(1-v**2/c**2)', 'Formula': 'm*c**2/np.sqrt(1-v**2/c**2)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman51', 'DescriptiveName': 'Feynman51, Lecture I.50.26', 'Formula_Str': 'x1*(cos(omega*t)+alpha*cos(omega*t)**2)', 'Formula': 'x1*(np.cos(omega*t)+alpha*np.cos(omega*t)**2)', 'Variables': [{'name': 'x1', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 3.0}, {'name': 't', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman52', 'DescriptiveName': 'Feynman52, Lecture II.2.42', 'Formula_Str': 'kappa*(T2-T1)*A/d', 'Formula': 'kappa*(T2-T1)*A/d', 'Variables': [{'name': 'kappa', 'low': 1.0, 'high': 5.0}, {'name': 'T1', 'low': 1.0, 'high': 5.0}, {'name': 'T2', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman53', 'DescriptiveName': 'Feynman53, Lecture II.3.24', 'Formula_Str': 'Pwr/(4*pi*r**2)', 'Formula': 'Pwr/(4*np.pi*r**2)', 'Variables': [{'name': 'Pwr', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman54', 'DescriptiveName': 'Feynman54, Lecture II.4.23', 'Formula_Str': 'q/(4*pi*epsilon*r)', 'Formula': 'q/(4*np.pi*epsilon*r)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman55', 'DescriptiveName': 'Feynman55, Lecture II.6.11', 'Formula_Str': '1/(4*pi*epsilon)*p_d*cos(theta)/r**2', 'Formula': '1/(4*np.pi*epsilon)*p_d*np.cos(theta)/r**2', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman56', 'DescriptiveName': 'Feynman56, Lecture II.6.15a', 'Formula_Str': 'p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)', 'Formula': 'p_d/(4*np.pi*epsilon)*3*z/r**5*np.sqrt(x**2+y**2)', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}, {'name': 'x', 'low': 1.0, 'high': 3.0}, {'name': 'y', 'low': 1.0, 'high': 3.0}, {'name': 'z', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman57', 'DescriptiveName': 'Feynman57, Lecture II.6.15b', 'Formula_Str': 'p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3', 'Formula': 'p_d/(4*np.pi*epsilon)*3*np.cos(theta)*np.sin(theta)/r**3', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman58', 'DescriptiveName': 'Feynman58, Lecture II.8.7', 'Formula_Str': '3/5*q**2/(4*pi*epsilon*d)', 'Formula': '3/5*q**2/(4*np.pi*epsilon*d)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman59', 'DescriptiveName': 'Feynman59, Lecture II.8.31', 'Formula_Str': 'epsilon*Ef**2/2', 'Formula': 'epsilon*Ef**2/2', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman60', 'DescriptiveName': 'Feynman60, Lecture II.10.9', 'Formula_Str': 'sigma_den/epsilon*1/(1+chi)', 'Formula': 'sigma_den/epsilon*1/(1+chi)', 'Variables': [{'name': 'sigma_den', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'chi', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman61', 'DescriptiveName': 'Feynman61, Lecture II.11.3', 'Formula_Str': 'q*Ef/(m*(omega_0**2-omega**2))', 'Formula': 'q*Ef/(m*(omega_0**2-omega**2))', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'omega_0', 'low': 3.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Feynman62', 'DescriptiveName': 'Feynman62, Lecture II.11.17', 'Formula_Str': 'n_0*(1+p_d*Ef*cos(theta)/(kb*T))', 'Formula': 'n_0*(1+p_d*Ef*np.cos(theta)/(kb*T))', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}, {'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman63', 'DescriptiveName': 'Feynman63, Lecture II.11.20', 'Formula_Str': 'n_rho*p_d**2*Ef/(3*kb*T)', 'Formula': 'n_rho*p_d**2*Ef/(3*kb*T)', 'Variables': [{'name': 'n_rho', 'low': 1.0, 'high': 5.0}, {'name': 'p_d', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman64', 'DescriptiveName': 'Feynman64, Lecture II.11.27', 'Formula_Str': 'n*alpha/(1-(n*alpha/3))*epsilon*Ef', 'Formula': 'n*alpha/(1-(n*alpha/3))*epsilon*Ef', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'alpha', 'low': 0.0, 'high': 1.0}, {'name': 'epsilon', 'low': 1.0, 'high': 2.0}, {'name': 'Ef', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Feynman65', 'DescriptiveName': 'Feynman65, Lecture II.11.28', 'Formula_Str': '1+n*alpha/(1-(n*alpha/3))', 'Formula': '1+n*alpha/(1-(n*alpha/3))', 'Variables': [{'name': 'n', 'low': 0.0, 'high': 1.0}, {'name': 'alpha', 'low': 0.0, 'high': 1.0}]},
{'FunctionName': 'Feynman66', 'DescriptiveName': 'Feynman66, Lecture II.13.17', 'Formula_Str': '1/(4*pi*epsilon*c**2)*2*I/r', 'Formula': '1/(4*np.pi*epsilon*c**2)*2*I/r', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'I', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman67', 'DescriptiveName': 'Feynman67, Lecture II.13.23', 'Formula_Str': 'rho_c_0/sqrt(1-v**2/c**2)', 'Formula': 'rho_c_0/np.sqrt(1-v**2/c**2)', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman68', 'DescriptiveName': 'Feynman68, Lecture II.13.34', 'Formula_Str': 'rho_c_0*v/sqrt(1-v**2/c**2)', 'Formula': 'rho_c_0*v/np.sqrt(1-v**2/c**2)', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman69', 'DescriptiveName': 'Feynman69, Lecture II.15.4', 'Formula_Str': '-mom*B*cos(theta)', 'Formula': '-mom*B*np.cos(theta)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman70', 'DescriptiveName': 'Feynman70, Lecture II.15.5', 'Formula_Str': '-p_d*Ef*cos(theta)', 'Formula': '-p_d*Ef*np.cos(theta)', 'Variables': [{'name': 'p_d', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman71', 'DescriptiveName': 'Feynman71, Lecture II.21.32', 'Formula_Str': 'q/(4*pi*epsilon*r*(1-v/c))', 'Formula': 'q/(4*np.pi*epsilon*r*(1-v/c))', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 3.0, 'high': 10.0}]},
{'FunctionName': 'Feynman72', 'DescriptiveName': 'Feynman72, Lecture II.24.17', 'Formula_Str': 'sqrt(omega**2/c**2-pi**2/d**2)', 'Formula': 'np.sqrt(omega**2/c**2-np.pi**2/d**2)', 'Variables': [{'name': 'omega', 'low': 4.0, 'high': 6.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 2.0, 'high': 4.0}]},
{'FunctionName': 'Feynman73', 'DescriptiveName': 'Feynman73, Lecture II.27.16', 'Formula_Str': 'epsilon*c*Ef**2', 'Formula': 'epsilon*c*Ef**2', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman74', 'DescriptiveName': 'Feynman74, Lecture II.27.18', 'Formula_Str': 'epsilon*Ef**2', 'Formula': 'epsilon*Ef**2', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 5.0}, {'name': 'Ef', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman75', 'DescriptiveName': 'Feynman75, Lecture II.34.2a', 'Formula_Str': 'q*v/(2*pi*r)', 'Formula': 'q*v/(2*np.pi*r)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman76', 'DescriptiveName': 'Feynman76, Lecture II.34.2', 'Formula_Str': 'q*v*r/2', 'Formula': 'q*v*r/2', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'v', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman77', 'DescriptiveName': 'Feynman77, Lecture II.34.11', 'Formula_Str': 'g_*q*B/(2*m)', 'Formula': 'g_*q*B/(2*m)', 'Variables': [{'name': 'g_', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman78', 'DescriptiveName': 'Feynman78, Lecture II.34.29a', 'Formula_Str': 'q*h/(4*pi*m)', 'Formula': 'q*h/(4*np.pi*m)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman79', 'DescriptiveName': 'Feynman79, Lecture II.34.29b', 'Formula_Str': 'g_*mom*B*Jz/(h/(2*pi))', 'Formula': 'g_*mom*B*Jz/(h/(2*np.pi))', 'Variables': [{'name': 'g_', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'Jz', 'low': 1.0, 'high': 5.0}, {'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman80', 'DescriptiveName': 'Feynman80, Lecture II.35.18', 'Formula_Str': 'n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))', 'Formula': 'n_0/(np.exp(mom*B/(kb*T))+np.exp(-mom*B/(kb*T)))', 'Variables': [{'name': 'n_0', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'mom', 'low': 1.0, 'high': 3.0}, {'name': 'B', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman81', 'DescriptiveName': 'Feynman81, Lecture II.35.21', 'Formula_Str': 'n_rho*mom*tanh(mom*B/(kb*T))', 'Formula': 'n_rho*mom*np.tanh(mom*B/(kb*T))', 'Variables': [{'name': 'n_rho', 'low': 1.0, 'high': 5.0}, {'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman82', 'DescriptiveName': 'Feynman82, Lecture II.36.38', 'Formula_Str': 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M', 'Formula': 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 3.0}, {'name': 'H', 'low': 1.0, 'high': 3.0}, {'name': 'kb', 'low': 1.0, 'high': 3.0}, {'name': 'T', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}, {'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'c', 'low': 1.0, 'high': 3.0}, {'name': 'M', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Feynman83', 'DescriptiveName': 'Feynman83, Lecture II.37.1', 'Formula_Str': 'mom*(1+chi)*B', 'Formula': 'mom*(1+chi)*B', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'chi', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman84', 'DescriptiveName': 'Feynman84, Lecture II.38.3', 'Formula_Str': 'Y*A*x/d', 'Formula': 'Y*A*x/d', 'Variables': [{'name': 'Y', 'low': 1.0, 'high': 5.0}, {'name': 'A', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman85', 'DescriptiveName': 'Feynman85, Lecture II.38.14', 'Formula_Str': 'Y/(2*(1+sigma))', 'Formula': 'Y/(2*(1+sigma))', 'Variables': [{'name': 'Y', 'low': 1.0, 'high': 5.0}, {'name': 'sigma', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman86', 'DescriptiveName': 'Feynman86, Lecture III.4.32', 'Formula_Str': '1/(exp((h/(2*pi))*omega/(kb*T))-1)', 'Formula': '1/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman87', 'DescriptiveName': 'Feynman87, Lecture III.4.33', 'Formula_Str': '(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)', 'Formula': '(h/(2*np.pi))*omega/(np.exp((h/(2*np.pi))*omega/(kb*T))-1)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'kb', 'low': 1.0, 'high': 5.0}, {'name': 'T', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman88', 'DescriptiveName': 'Feynman88, Lecture III.7.38', 'Formula_Str': '2*mom*B/(h/(2*pi))', 'Formula': '2*mom*B/(h/(2*np.pi))', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'B', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman89', 'DescriptiveName': 'Feynman89, Lecture III.8.54', 'Formula_Str': 'sin(E_n*t/(h/(2*pi)))**2', 'Formula': 'np.sin(E_n*t/(h/(2*np.pi)))**2', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 2.0}, {'name': 't', 'low': 1.0, 'high': 2.0}, {'name': 'h', 'low': 1.0, 'high': 4.0}]},
{'FunctionName': 'Feynman90', 'DescriptiveName': 'Feynman90, Lecture III.9.52', 'Formula_Str': '(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2', 'Formula': '(p_d*Ef*t/(h/(2*np.pi)))*np.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2', 'Variables': [{'name': 'p_d', 'low': 1.0, 'high': 3.0}, {'name': 'Ef', 'low': 1.0, 'high': 3.0}, {'name': 't', 'low': 1.0, 'high': 3.0}, {'name': 'h', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman91', 'DescriptiveName': 'Feynman91, Lecture III.10.19', 'Formula_Str': 'mom*sqrt(Bx**2+By**2+Bz**2)', 'Formula': 'mom*np.sqrt(Bx**2+By**2+Bz**2)', 'Variables': [{'name': 'mom', 'low': 1.0, 'high': 5.0}, {'name': 'Bx', 'low': 1.0, 'high': 5.0}, {'name': 'By', 'low': 1.0, 'high': 5.0}, {'name': 'Bz', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman92', 'DescriptiveName': 'Feynman92, Lecture III.12.43', 'Formula_Str': 'n*(h/(2*pi))', 'Formula': 'n*(h/(2*np.pi))', 'Variables': [{'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman93', 'DescriptiveName': 'Feynman93, Lecture III.13.18', 'Formula_Str': '2*E_n*d**2*k/(h/(2*pi))', 'Formula': '2*E_n*d**2*k/(h/(2*np.pi))', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'k', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman94', 'DescriptiveName': 'Feynman94, Lecture III.14.14', 'Formula_Str': 'I_0*(exp(q*Volt/(kb*T))-1)', 'Formula': 'I_0*(np.exp(q*Volt/(kb*T))-1)', 'Variables': [{'name': 'I_0', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 2.0}, {'name': 'Volt', 'low': 1.0, 'high': 2.0}, {'name': 'kb', 'low': 1.0, 'high': 2.0}, {'name': 'T', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Feynman95', 'DescriptiveName': 'Feynman95, Lecture III.15.12', 'Formula_Str': '2*U*(1-cos(k*d))', 'Formula': '2*U*(1-np.cos(k*d))', 'Variables': [{'name': 'U', 'low': 1.0, 'high': 5.0}, {'name': 'k', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman96', 'DescriptiveName': 'Feynman96, Lecture III.15.14', 'Formula_Str': '(h/(2*pi))**2/(2*E_n*d**2)', 'Formula': '(h/(2*np.pi))**2/(2*E_n*d**2)', 'Variables': [{'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'E_n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman97', 'DescriptiveName': 'Feynman97, Lecture III.15.27', 'Formula_Str': '2*pi*alpha/(n*d)', 'Formula': '2*np.pi*alpha/(n*d)', 'Variables': [{'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman98', 'DescriptiveName': 'Feynman98, Lecture III.17.37', 'Formula_Str': 'beta*(1+alpha*cos(theta))', 'Formula': 'beta*(1+alpha*np.cos(theta))', 'Variables': [{'name': 'beta', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman99', 'DescriptiveName': 'Feynman99, Lecture III.19.51', 'Formula_Str': '-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)', 'Formula': '-m*q**4/(2*(4*np.pi*epsilon)**2*(h/(2*np.pi))**2)*(1/n**2)', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'n', 'low': 1.0, 'high': 5.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Feynman100', 'DescriptiveName': 'Feynman100, Lecture III.21.20', 'Formula_Str': '-rho_c_0*q*A_vec/m', 'Formula': '-rho_c_0*q*A_vec/m', 'Variables': [{'name': 'rho_c_0', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'A_vec', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus1', 'DescriptiveName': 'Bonus1.0, Rutherford scattering', 'Formula_Str': '(Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2', 'Formula': '(Z_1*Z_2*alpha*hbar*c/(4*E_n*np.sin(theta/2)**2))**2', 'Variables': [{'name': 'Z_1', 'low': 1.0, 'high': 2.0}, {'name': 'Z_2', 'low': 1.0, 'high': 2.0}, {'name': 'alpha', 'low': 1.0, 'high': 2.0}, {'name': 'hbar', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus2', 'DescriptiveName': 'Bonus2.0, 3.55 Goldstein', 'Formula_Str': 'm*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))', 'Formula': 'm*k_G/L**2*(1+np.sqrt(1+2*E_n*L**2/(m*k_G**2))*np.cos(theta1-theta2))', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'k_G', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'theta1', 'low': 0.0, 'high': 6.0}, {'name': 'theta2', 'low': 0.0, 'high': 6.0}]},
{'FunctionName': 'Bonus3', 'DescriptiveName': 'Bonus3.0, 3.64 Goldstein', 'Formula_Str': 'd*(1-alpha**2)/(1+alpha*cos(theta1-theta2))', 'Formula': 'd*(1-alpha**2)/(1+alpha*np.cos(theta1-theta2))', 'Variables': [{'name': 'd', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 2.0, 'high': 4.0}, {'name': 'theta1', 'low': 4.0, 'high': 5.0}, {'name': 'theta2', 'low': 4.0, 'high': 5.0}]},
{'FunctionName': 'Bonus4', 'DescriptiveName': 'Bonus4.0, 3.16 Goldstein', 'Formula_Str': 'sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))', 'Formula': 'np.sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 8.0, 'high': 12.0}, {'name': 'U', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus5', 'DescriptiveName': 'Bonus5.0, 3.74 Goldstein', 'Formula_Str': '2*pi*d**(3/2)/sqrt(G*(m1+m2))', 'Formula': '2*np.pi*d**(3/2)/np.sqrt(G*(m1+m2))', 'Variables': [{'name': 'd', 'low': 1.0, 'high': 3.0}, {'name': 'G', 'low': 1.0, 'high': 3.0}, {'name': 'm1', 'low': 1.0, 'high': 3.0}, {'name': 'm2', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus6', 'DescriptiveName': 'Bonus6.0, 3.99 Goldstein', 'Formula_Str': 'sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))', 'Formula': 'np.sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))', 'Variables': [{'name': 'epsilon', 'low': 1.0, 'high': 3.0}, {'name': 'L', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'Z_1', 'low': 1.0, 'high': 3.0}, {'name': 'Z_2', 'low': 1.0, 'high': 3.0}, {'name': 'q', 'low': 1.0, 'high': 3.0}, {'name': 'E_n', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus7', 'DescriptiveName': 'Bonus7.0, Friedman Equation', 'Formula_Str': 'sqrt(8*pi*G*rho/3-alpha*c**2/d**2)', 'Formula': 'np.sqrt(8*np.pi*G*rho/3-alpha*c**2/d**2)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 3.0}, {'name': 'rho', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'd', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus8', 'DescriptiveName': 'Bonus8.0, Compton Scattering', 'Formula_Str': 'E_n/(1+E_n/(m*c**2)*(1-cos(theta)))', 'Formula': 'E_n/(1+E_n/(m*c**2)*(1-np.cos(theta)))', 'Variables': [{'name': 'E_n', 'low': 1.0, 'high': 3.0}, {'name': 'm', 'low': 1.0, 'high': 3.0}, {'name': 'c', 'low': 1.0, 'high': 3.0}, {'name': 'theta', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus9', 'DescriptiveName': 'Bonus9.0, Gravitational wave ratiated power', 'Formula_Str': '-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5', 'Formula': '-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 2.0}, {'name': 'c', 'low': 1.0, 'high': 2.0}, {'name': 'm1', 'low': 1.0, 'high': 5.0}, {'name': 'm2', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Bonus10', 'DescriptiveName': 'Bonus10.0, Relativistic aberation', 'Formula_Str': 'arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))', 'Formula': 'np.arccos((np.cos(theta2)-v/c)/(1-v/c*np.cos(theta2)))', 'Variables': [{'name': 'c', 'low': 4.0, 'high': 6.0}, {'name': 'v', 'low': 1.0, 'high': 3.0}, {'name': 'theta2', 'low': 1.0, 'high': 3.0}]},
{'FunctionName': 'Bonus11', 'DescriptiveName': 'Bonus11.0, N-slit diffraction', 'Formula_Str': 'I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2', 'Formula': 'I_0*(np.sin(alpha/2)*np.sin(n*delta/2)/(alpha/2*np.sin(delta/2)))**2', 'Variables': [{'name': 'I_0', 'low': 1.0, 'high': 3.0}, {'name': 'alpha', 'low': 1.0, 'high': 3.0}, {'name': 'delta', 'low': 1.0, 'high': 3.0}, {'name': 'n', 'low': 1.0, 'high': 2.0}]},
{'FunctionName': 'Bonus12', 'DescriptiveName': 'Bonus12.0, 2.11 Jackson', 'Formula_Str': 'q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)', 'Formula': 'q/(4*np.pi*epsilon*y**2)*(4*np.pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'y', 'low': 1.0, 'high': 3.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 4.0, 'high': 6.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus13', 'DescriptiveName': 'Bonus13.0, 3.45 Jackson', 'Formula_Str': '1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))', 'Formula': '1/(4*np.pi*epsilon)*q/np.sqrt(r**2+d**2-2*r*d*np.cos(alpha))', 'Variables': [{'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 3.0}, {'name': 'd', 'low': 4.0, 'high': 6.0}, {'name': 'alpha', 'low': 0.0, 'high': 6.0}, {'name': 'epsilon', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus14', 'DescriptiveName': "Bonus14.0, 4.60' Jackson", 'Formula_Str': 'Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))', 'Formula': 'Ef*np.cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))', 'Variables': [{'name': 'Ef', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 6.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'd', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus15', 'DescriptiveName': 'Bonus15.0, 11.38 Jackson', 'Formula_Str': 'sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))', 'Formula': 'np.sqrt(1-v**2/c**2)*omega/(1+v/c*np.cos(theta))', 'Variables': [{'name': 'c', 'low': 5.0, 'high': 20.0}, {'name': 'v', 'low': 1.0, 'high': 3.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'theta', 'low': 0.0, 'high': 6.0}]},
{'FunctionName': 'Bonus16', 'DescriptiveName': 'Bonus16.0, 8.56 Goldstein', 'Formula_Str': 'sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt', 'Formula': 'np.sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}, {'name': 'q', 'low': 1.0, 'high': 5.0}, {'name': 'A_vec', 'low': 1.0, 'high': 5.0}, {'name': 'Volt', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus17', 'DescriptiveName': "Bonus17.0, 12.80' Goldstein", 'Formula_Str': '1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))', 'Formula': '1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))', 'Variables': [{'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'p', 'low': 1.0, 'high': 5.0}, {'name': 'y', 'low': 1.0, 'high': 5.0}, {'name': 'x', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus18', 'DescriptiveName': 'Bonus18.0, 15.2.1 Weinberg', 'Formula_Str': '3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)', 'Formula': '3/(8*np.pi*G)*(c**2*k_f/r**2+H_G**2)', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 5.0}, {'name': 'k_f', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'H_G', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus19', 'DescriptiveName': 'Bonus19.0, 15.2.2 Weinberg', 'Formula_Str': '-1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))', 'Formula': '-1/(8*np.pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))', 'Variables': [{'name': 'G', 'low': 1.0, 'high': 5.0}, {'name': 'k_f', 'low': 1.0, 'high': 5.0}, {'name': 'r', 'low': 1.0, 'high': 5.0}, {'name': 'H_G', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}]},
{'FunctionName': 'Bonus20', 'DescriptiveName': 'Bonus20.0, Klein-Nishina (13.132 Schwarz)', 'Formula_Str': '1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)', 'Formula': '1/(4*np.pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-np.sin(beta)**2)', 'Variables': [{'name': 'omega', 'low': 1.0, 'high': 5.0}, {'name': 'omega_0', 'low': 1.0, 'high': 5.0}, {'name': 'alpha', 'low': 1.0, 'high': 5.0}, {'name': 'h', 'low': 1.0, 'high': 5.0}, {'name': 'm', 'low': 1.0, 'high': 5.0}, {'name': 'c', 'low': 1.0, 'high': 5.0}, {'name': 'beta', 'low': 0.0, 'high': 6.0}]} 
 ]