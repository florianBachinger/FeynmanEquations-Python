constraints = [
  {
    'EquationName': 'Feynman1',
    'DescriptiveName': 'Feynman1, Lecture I.6.2a',
    'Constraints': [
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*theta*exp(-theta**2/2)/(2*sqrt(pi))',
        'derivative_lambda': 'lambda args : (lambda theta: -np.sqrt(2)*theta*np.exp(-theta**2/2)/(2*np.sqrt(np.pi)) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(2)*(theta**2 - 1)*exp(-theta**2/2)/(2*sqrt(pi))',
        'derivative_lambda': 'lambda args : (lambda theta: np.sqrt(2)*(theta**2 - 1)*np.exp(-theta**2/2)/(2*np.sqrt(np.pi)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman2',
    'DescriptiveName': 'Feynman2, Lecture I.6.2',
    'Constraints': [
      {
        'name': 'sigma',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*exp(-theta**2/(2*sigma**2))/(2*sqrt(pi)*sigma**2) + sqrt(2)*theta**2*exp(-theta**2/(2*sigma**2))/(2*sqrt(pi)*sigma**4)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta: -np.sqrt(2)*np.exp(-theta**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**2) + np.sqrt(2)*theta**2*np.exp(-theta**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**4) )(*args)'
      },
      {
        'name': 'sigma',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'sqrt(2)*(1 - theta**2*(3 - theta**2/sigma**2)/(2*sigma**2) - theta**2/sigma**2)*exp(-theta**2/(2*sigma**2))/(sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta: np.sqrt(2)*(1 - theta**2*(3 - theta**2/sigma**2)/(2*sigma**2) - theta**2/sigma**2)*np.exp(-theta**2/(2*sigma**2))/(np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*theta*exp(-theta**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta: -np.sqrt(2)*theta*np.exp(-theta**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*(1 - theta**2/sigma**2)*exp(-theta**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta: -np.sqrt(2)*(1 - theta**2/sigma**2)*np.exp(-theta**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'sigma',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman3',
    'DescriptiveName': 'Feynman3, Lecture I.6.2b',
    'Constraints': [
      {
        'name': 'sigma',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*exp(-(theta - theta1)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**2) + sqrt(2)*(theta - theta1)**2*exp(-(theta - theta1)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**4)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: -np.sqrt(2)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**2) + np.sqrt(2)*(theta - theta1)**2*np.exp(-(theta - theta1)**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**4) )(*args)'
      },
      {
        'name': 'sigma',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'sqrt(2)*(1 - (3 - (theta - theta1)**2/sigma**2)*(theta - theta1)**2/(2*sigma**2) - (theta - theta1)**2/sigma**2)*exp(-(theta - theta1)**2/(2*sigma**2))/(sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: np.sqrt(2)*(1 - (3 - (theta - theta1)**2/sigma**2)*(theta - theta1)**2/(2*sigma**2) - (theta - theta1)**2/sigma**2)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*(2*theta - 2*theta1)*exp(-(theta - theta1)**2/(2*sigma**2))/(4*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: -np.sqrt(2)*(2*theta - 2*theta1)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(4*np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*(1 - (theta - theta1)**2/sigma**2)*exp(-(theta - theta1)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: -np.sqrt(2)*(1 - (theta - theta1)**2/sigma**2)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*(-2*theta + 2*theta1)*exp(-(theta - theta1)**2/(2*sigma**2))/(4*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: -np.sqrt(2)*(-2*theta + 2*theta1)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(4*np.sqrt(np.pi)*sigma**3) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-sqrt(2)*(1 - (theta - theta1)**2/sigma**2)*exp(-(theta - theta1)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**3)',
        'derivative_lambda': 'lambda args : (lambda sigma,theta,theta1: -np.sqrt(2)*(1 - (theta - theta1)**2/sigma**2)*np.exp(-(theta - theta1)**2/(2*sigma**2))/(2*np.sqrt(np.pi)*sigma**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'sigma',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta1',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman4',
    'DescriptiveName': 'Feynman4, Lecture I.8.14',
    'Constraints': [
      {
        'name': 'x1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(x1 - x2)/sqrt((-x1 + x2)**2 + (-y1 + y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (x1 - x2)/np.sqrt((-x1 + x2)**2 + (-y1 + y2)**2) )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/sqrt((x1 - x2)**2 + (y1 - y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2) )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(-x1 + x2)/sqrt((-x1 + x2)**2 + (-y1 + y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-x1 + x2)/np.sqrt((-x1 + x2)**2 + (-y1 + y2)**2) )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/sqrt((x1 - x2)**2 + (y1 - y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2) )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(y1 - y2)/sqrt((-x1 + x2)**2 + (-y1 + y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (y1 - y2)/np.sqrt((-x1 + x2)**2 + (-y1 + y2)**2) )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/sqrt((x1 - x2)**2 + (y1 - y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2) )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(-y1 + y2)/sqrt((-x1 + x2)**2 + (-y1 + y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-y1 + y2)/np.sqrt((-x1 + x2)**2 + (-y1 + y2)**2) )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/sqrt((x1 - x2)**2 + (y1 - y2)**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,y1,y2: (-(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2) + 1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y2',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman5',
    'DescriptiveName': 'Feynman5, Lecture I.9.18',
    'Constraints': [
      {
        'name': 'm1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m2/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m2/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 0 )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m1/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 0 )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm1*m2/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: m1*m2/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 0 )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'G*m1*m2*(-2*x1 + 2*x2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(-2*x1 + 2*x2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m1*m2*(2*x1 - 2*x2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(2*x1 - 2*x2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(x1 - x2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'G*m1*m2*(-2*y1 + 2*y2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(-2*y1 + 2*y2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m1*m2*(2*y1 - 2*y2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(2*y1 - 2*y2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(y1 - y2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      },
      {
        'name': 'z1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'G*m1*m2*(-2*z1 + 2*z2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(-2*z1 + 2*z2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'z1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(z1 - z2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(z1 - z2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      },
      {
        'name': 'z2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m1*m2*(2*z1 - 2*z2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: G*m1*m2*(2*z1 - 2*z2)/((-x1 + x2)**2 + (-y1 + y2)**2 + (-z1 + z2)**2)**2 )(*args)'
      },
      {
        'name': 'z2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*G*m1*m2*(4*(z1 - z2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2: 2*G*m1*m2*(4*(z1 - z2)**2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) - 1)/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm1',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'm2',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'G',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'x1',
        'low': 3.0,
        'high': 4.0
      },
      {
        'name': 'x2',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'y1',
        'low': 3.0,
        'high': 4.0
      },
      {
        'name': 'y2',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'z1',
        'low': 3.0,
        'high': 4.0
      },
      {
        'name': 'z2',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Feynman6',
    'DescriptiveName': 'Feynman6, Lecture I.10.7',
    'Constraints': [
      {
        'name': 'm_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: 1/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'm_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm_0*v/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: m_0*v/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm_0*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: m_0*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-m_0*v**2/(c**3*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: -m_0*v**2/(c**3*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*m_0*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: 3*m_0*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman7',
    'DescriptiveName': 'Feynman7, Lecture I.11.19',
    'Constraints': [
      {
        'name': 'x1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'y1',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: y1 )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'y2',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: y2 )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      },
      {
        'name': 'x3',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'y3',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: y3 )(*args)'
      },
      {
        'name': 'x3',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x1',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: x1 )(*args)'
      },
      {
        'name': 'y1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x2',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: x2 )(*args)'
      },
      {
        'name': 'y2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      },
      {
        'name': 'y3',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x3',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: x3 )(*args)'
      },
      {
        'name': 'y3',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,x2,x3,y1,y2,y3: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x3',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y3',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman8',
    'DescriptiveName': 'Feynman8, Lecture I.12.1',
    'Constraints': [
      {
        'name': 'mu',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Nn',
        'derivative_lambda': 'lambda args : (lambda mu,Nn: Nn )(*args)'
      },
      {
        'name': 'mu',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mu,Nn: 0 )(*args)'
      },
      {
        'name': 'Nn',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'mu',
        'derivative_lambda': 'lambda args : (lambda mu,Nn: mu )(*args)'
      },
      {
        'name': 'Nn',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mu,Nn: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mu',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Nn',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman10',
    'DescriptiveName': 'Feynman10, Lecture I.12.2',
    'Constraints': [
      {
        'name': 'q1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q2/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: q2/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'q1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: 0 )(*args)'
      },
      {
        'name': 'q2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q1/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: q1/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'q2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q1*q2/(4*pi*epsilon**2*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: -q1*q2/(4*np.pi*epsilon**2*r**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q1*q2/(2*pi*epsilon**3*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: q1*q2/(2*np.pi*epsilon**3*r**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q1*q2/(2*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: -q1*q2/(2*np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*q1*q2/(2*pi*epsilon*r**4)',
        'derivative_lambda': 'lambda args : (lambda q1,q2,epsilon,r: 3*q1*q2/(2*np.pi*epsilon*r**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman11',
    'DescriptiveName': 'Feynman11, Lecture I.12.4',
    'Constraints': [
      {
        'name': 'q1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: 1/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'q1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q1/(4*pi*epsilon**2*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: -q1/(4*np.pi*epsilon**2*r**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q1/(2*pi*epsilon**3*r**2)',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: q1/(2*np.pi*epsilon**3*r**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q1/(2*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: -q1/(2*np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*q1/(2*pi*epsilon*r**4)',
        'derivative_lambda': 'lambda args : (lambda q1,epsilon,r: 3*q1/(2*np.pi*epsilon*r**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman12',
    'DescriptiveName': 'Feynman12, Lecture I.12.5',
    'Constraints': [
      {
        'name': 'q2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef',
        'derivative_lambda': 'lambda args : (lambda q2,Ef: Ef )(*args)'
      },
      {
        'name': 'q2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q2,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q2',
        'derivative_lambda': 'lambda args : (lambda q2,Ef: q2 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q2,Ef: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman13',
    'DescriptiveName': 'Feynman13, Lecture I.12.11',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'B*v*sin(theta) + Ef',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: B*v*np.sin(theta) + Ef )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: q )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'q*v*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: q*v*np.sin(theta) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'B*q*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: B*q*np.sin(theta) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'B*q*v*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: B*q*v*np.cos(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-B*q*v*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,B,v,theta: -B*q*v*np.sin(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman9',
    'DescriptiveName': 'Feynman9, Lecture I.13.4',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'u**2/2 + v**2/2 + w**2/2',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: u**2/2 + v**2/2 + w**2/2 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*v',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m*v )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*u',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m*u )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m )(*args)'
      },
      {
        'name': 'w',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*w',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m*w )(*args)'
      },
      {
        'name': 'w',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm',
        'derivative_lambda': 'lambda args : (lambda m,v,u,w: m )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'u',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'w',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman14',
    'DescriptiveName': 'Feynman14, Lecture I.13.12',
    'Constraints': [
      {
        'name': 'm1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'G*m2*(1/r2 - 1/r1)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: G*m2*(1/r2 - 1/r1) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: 0 )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'G*m1*(1/r2 - 1/r1)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: G*m1*(1/r2 - 1/r1) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: 0 )(*args)'
      },
      {
        'name': 'r1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'G*m1*m2/r1**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: G*m1*m2/r1**2 )(*args)'
      },
      {
        'name': 'r1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*G*m1*m2/r1**3',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: -2*G*m1*m2/r1**3 )(*args)'
      },
      {
        'name': 'r2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-G*m1*m2/r2**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: -G*m1*m2/r2**2 )(*args)'
      },
      {
        'name': 'r2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*G*m1*m2/r2**3',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: 2*G*m1*m2/r2**3 )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'm1*m2*(1/r2 - 1/r1)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: m1*m2*(1/r2 - 1/r1) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2,G: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'G',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman15',
    'DescriptiveName': 'Feynman15, Lecture I.14.3',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g*z',
        'derivative_lambda': 'lambda args : (lambda m,g,z: g*z )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,g,z: 0 )(*args)'
      },
      {
        'name': 'g',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*z',
        'derivative_lambda': 'lambda args : (lambda m,g,z: m*z )(*args)'
      },
      {
        'name': 'g',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,g,z: 0 )(*args)'
      },
      {
        'name': 'z',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g*m',
        'derivative_lambda': 'lambda args : (lambda m,g,z: g*m )(*args)'
      },
      {
        'name': 'z',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,g,z: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'g',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'z',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman16',
    'DescriptiveName': 'Feynman16, Lecture I.14.4',
    'Constraints': [
      {
        'name': 'k_spring',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x**2/2',
        'derivative_lambda': 'lambda args : (lambda k_spring,x: x**2/2 )(*args)'
      },
      {
        'name': 'k_spring',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda k_spring,x: 0 )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'k_spring*x',
        'derivative_lambda': 'lambda args : (lambda k_spring,x: k_spring*x )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'k_spring',
        'derivative_lambda': 'lambda args : (lambda k_spring,x: k_spring )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'k_spring',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman17',
    'DescriptiveName': 'Feynman17, Lecture I.15.3x',
    'Constraints': [
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/sqrt(1 - u**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: 1/np.sqrt(1 - u**2/c**2) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: 0 )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-t/sqrt(1 - u**2/c**2) + u*(-t*u + x)/(c**2*(1 - u**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: -t/np.sqrt(1 - u**2/c**2) + u*(-t*u + x)/(c**2*(1 - u**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-(2*t*u + (1 + 3*u**2/(c**2*(1 - u**2/c**2)))*(t*u - x))/(c**2*(1 - u**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: -(2*t*u + (1 + 3*u**2/(c**2*(1 - u**2/c**2)))*(t*u - x))/(c**2*(1 - u**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-u**2*(-t*u + x)/(c**3*(1 - u**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: -u**2*(-t*u + x)/(c**3*(1 - u**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-3*u**2*(1 + u**2/(c**2*(1 - u**2/c**2)))*(t*u - x)/(c**4*(1 - u**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: -3*u**2*(1 + u**2/(c**2*(1 - u**2/c**2)))*(t*u - x)/(c**4*(1 - u**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-u/sqrt(1 - u**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: -u/np.sqrt(1 - u**2/c**2) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x,u,c,t: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x',
        'low': 5.0,
        'high': 10.0
      },
      {
        'name': 'u',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 20.0
      },
      {
        'name': 't',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Feynman18',
    'DescriptiveName': 'Feynman18, Lecture I.15.3t',
    'Constraints': [
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-u/(c**2*sqrt(1 - u**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: -u/(c**2*np.sqrt(1 - u**2/c**2)) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: 0 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-u**2*(t - u*x/c**2)/(c**3*(1 - u**2/c**2)**(3/2)) + 2*u*x/(c**3*sqrt(1 - u**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: -u**2*(t - u*x/c**2)/(c**3*(1 - u**2/c**2)**(3/2)) + 2*u*x/(c**3*np.sqrt(1 - u**2/c**2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'u*(3*u*(1 + u**2/(c**2*(1 - u**2/c**2)))*(t - u*x/c**2)/(1 - u**2/c**2) - 6*x - 4*u**2*x/(c**2*(1 - u**2/c**2)))/(c**4*sqrt(1 - u**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: u*(3*u*(1 + u**2/(c**2*(1 - u**2/c**2)))*(t - u*x/c**2)/(1 - u**2/c**2) - 6*x - 4*u**2*x/(c**2*(1 - u**2/c**2)))/(c**4*np.sqrt(1 - u**2/c**2)) )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'u*(t - u*x/c**2)/(c**2*(1 - u**2/c**2)**(3/2)) - x/(c**2*sqrt(1 - u**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: u*(t - u*x/c**2)/(c**2*(1 - u**2/c**2)**(3/2)) - x/(c**2*np.sqrt(1 - u**2/c**2)) )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '((1 + 3*u**2/(c**2*(1 - u**2/c**2)))*(t - u*x/c**2) - 2*u*x/c**2)/(c**2*(1 - u**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: ((1 + 3*u**2/(c**2*(1 - u**2/c**2)))*(t - u*x/c**2) - 2*u*x/c**2)/(c**2*(1 - u**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/sqrt(1 - u**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: 1/np.sqrt(1 - u**2/c**2) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x,c,u,t: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      },
      {
        'name': 'u',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 't',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman19',
    'DescriptiveName': 'Feynman19, Lecture I.15.1',
    'Constraints': [
      {
        'name': 'm_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'v/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: v/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'm_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm_0/sqrt(1 - v**2/c**2) + m_0*v**2/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: m_0/np.sqrt(1 - v**2/c**2) + m_0*v**2/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm_0*v*(3 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: m_0*v*(3 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-m_0*v**3/(c**3*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: -m_0*v**3/(c**3*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*m_0*v**3*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m_0,v,c: 3*m_0*v**3*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman20',
    'DescriptiveName': 'Feynman20, Lecture I.16.6',
    'Constraints': [
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*u*v*(u + v)/(c**3*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: 2*u*v*(u + v)/(c**3*(1 + u*v/c**2)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*u*v*(3 - 4*u*v/(c**2*(1 + u*v/c**2)))*(u + v)/(c**4*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: -2*u*v*(3 - 4*u*v/(c**2*(1 + u*v/c**2)))*(u + v)/(c**4*(1 + u*v/c**2)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '1/(1 + u*v/c**2) - u*(u + v)/(c**2*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: 1/(1 + u*v/c**2) - u*(u + v)/(c**2*(1 + u*v/c**2)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*u*(-1 + u*(u + v)/(c**2*(1 + u*v/c**2)))/(c**2*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: 2*u*(-1 + u*(u + v)/(c**2*(1 + u*v/c**2)))/(c**2*(1 + u*v/c**2)**2) )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '1/(1 + u*v/c**2) - v*(u + v)/(c**2*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: 1/(1 + u*v/c**2) - v*(u + v)/(c**2*(1 + u*v/c**2)**2) )(*args)'
      },
      {
        'name': 'u',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*v*(-1 + v*(u + v)/(c**2*(1 + u*v/c**2)))/(c**2*(1 + u*v/c**2)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,u: 2*v*(-1 + v*(u + v)/(c**2*(1 + u*v/c**2)))/(c**2*(1 + u*v/c**2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'u',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman21',
    'DescriptiveName': 'Feynman21, Lecture I.18.4',
    'Constraints': [
      {
        'name': 'm1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'r1/(m1 + m2) - (m1*r1 + m2*r2)/(m1 + m2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: r1/(m1 + m2) - (m1*r1 + m2*r2)/(m1 + m2)**2 )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*(-r1 + (m1*r1 + m2*r2)/(m1 + m2))/(m1 + m2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: 2*(-r1 + (m1*r1 + m2*r2)/(m1 + m2))/(m1 + m2)**2 )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'r2/(m1 + m2) - (m1*r1 + m2*r2)/(m1 + m2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: r2/(m1 + m2) - (m1*r1 + m2*r2)/(m1 + m2)**2 )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*(-r2 + (m1*r1 + m2*r2)/(m1 + m2))/(m1 + m2)**2',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: 2*(-r2 + (m1*r1 + m2*r2)/(m1 + m2))/(m1 + m2)**2 )(*args)'
      },
      {
        'name': 'r1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm1/(m1 + m2)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: m1/(m1 + m2) )(*args)'
      },
      {
        'name': 'r1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: 0 )(*args)'
      },
      {
        'name': 'r2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm2/(m1 + m2)',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: m2/(m1 + m2) )(*args)'
      },
      {
        'name': 'r2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m1,m2,r1,r2: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r2',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman22',
    'DescriptiveName': 'Feynman22, Lecture I.18.12',
    'Constraints': [
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'F*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: F*np.sin(theta) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: 0 )(*args)'
      },
      {
        'name': 'F',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'r*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: r*np.sin(theta) )(*args)'
      },
      {
        'name': 'F',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'F*r*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: F*r*np.cos(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-F*r*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda r,F,theta: -F*r*np.sin(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'F',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 0.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman23',
    'DescriptiveName': 'Feynman23, Lecture I.18.14',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'r*v*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: r*v*np.sin(theta) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'm*v*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: m*v*np.sin(theta) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'm*r*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: m*r*np.sin(theta) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'm*r*v*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: m*r*v*np.cos(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-m*r*v*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda m,r,v,theta: -m*r*v*np.sin(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman24',
    'DescriptiveName': 'Feynman24, Lecture I.24.6',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x**2*(omega**2 + omega_0**2)/4',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: x**2*(omega**2 + omega_0**2)/4 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: 0 )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*omega*x**2/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*omega*x**2/2 )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*x**2/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*x**2/2 )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*omega_0*x**2/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*omega_0*x**2/2 )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*x**2/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*x**2/2 )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*x*(omega**2 + omega_0**2)/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*x*(omega**2 + omega_0**2)/2 )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*(omega**2 + omega_0**2)/2',
        'derivative_lambda': 'lambda args : (lambda m,omega,omega_0,x: m*(omega**2 + omega_0**2)/2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega_0',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman25',
    'DescriptiveName': 'Feynman25, Lecture I.25.13',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/C',
        'derivative_lambda': 'lambda args : (lambda q,C: 1/C )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,C: 0 )(*args)'
      },
      {
        'name': 'C',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/C**2',
        'derivative_lambda': 'lambda args : (lambda q,C: -q/C**2 )(*args)'
      },
      {
        'name': 'C',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*q/C**3',
        'derivative_lambda': 'lambda args : (lambda q,C: 2*q/C**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'C',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman26',
    'DescriptiveName': 'Feynman26, Lecture I.26.2',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'sin(theta2)/sqrt(-n**2*sin(theta2)**2 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,theta2: np.sin(theta2)/np.sqrt(-n**2*np.sin(theta2)**2 + 1) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'n*sin(theta2)**3/(-n**2*sin(theta2)**2 + 1)**(3/2)',
        'derivative_lambda': 'lambda args : (lambda n,theta2: n*np.sin(theta2)**3/(-n**2*np.sin(theta2)**2 + 1)**(3/2) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'n*cos(theta2)/sqrt(-n**2*sin(theta2)**2 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,theta2: n*np.cos(theta2)/np.sqrt(-n**2*np.sin(theta2)**2 + 1) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'n*(n**2*cos(theta2)**2/(-n**2*sin(theta2)**2 + 1) - 1)*sin(theta2)/sqrt(-n**2*sin(theta2)**2 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,theta2: n*(n**2*np.cos(theta2)**2/(-n**2*np.sin(theta2)**2 + 1) - 1)*np.sin(theta2)/np.sqrt(-n**2*np.sin(theta2)**2 + 1) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 0.0,
        'high': 1.0
      },
      {
        'name': 'theta2',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman27',
    'DescriptiveName': 'Feynman27, Lecture I.27.6',
    'Constraints': [
      {
        'name': 'd1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(d1**2*(n/d2 + 1/d1)**2)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: 1/(d1**2*(n/d2 + 1/d1)**2) )(*args)'
      },
      {
        'name': 'd1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '2*(-1 + 1/(d1*(n/d2 + 1/d1)))/(d1**3*(n/d2 + 1/d1)**2)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: 2*(-1 + 1/(d1*(n/d2 + 1/d1)))/(d1**3*(n/d2 + 1/d1)**2) )(*args)'
      },
      {
        'name': 'd2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'n/(d2**2*(n/d2 + 1/d1)**2)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: n/(d2**2*(n/d2 + 1/d1)**2) )(*args)'
      },
      {
        'name': 'd2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '2*n*(-1 + n/(d2*(n/d2 + 1/d1)))/(d2**3*(n/d2 + 1/d1)**2)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: 2*n*(-1 + n/(d2*(n/d2 + 1/d1)))/(d2**3*(n/d2 + 1/d1)**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-1/(d2*(n/d2 + 1/d1)**2)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: -1/(d2*(n/d2 + 1/d1)**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2/(d2**2*(n/d2 + 1/d1)**3)',
        'derivative_lambda': 'lambda args : (lambda d1,d2,n: 2/(d2**2*(n/d2 + 1/d1)**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'd1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman28',
    'DescriptiveName': 'Feynman28, Lecture I.29.4',
    'Constraints': [
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/c',
        'derivative_lambda': 'lambda args : (lambda omega,c: 1/c )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda omega,c: 0 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-omega/c**2',
        'derivative_lambda': 'lambda args : (lambda omega,c: -omega/c**2 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*omega/c**3',
        'derivative_lambda': 'lambda args : (lambda omega,c: 2*omega/c**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'omega',
        'low': 1.0,
        'high': 10.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman29',
    'DescriptiveName': 'Feynman29, Lecture I.29.16',
    'Constraints': [
      {
        'name': 'x1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(x1 - x2*cos(theta1 - theta2))/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: (x1 - x2*np.cos(theta1 - theta2))/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(x1 - x2*cos(theta1 - theta2))**2/(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2) + 1)/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: (-(x1 - x2*np.cos(theta1 - theta2))**2/(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) + 1)/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(-x1*cos(theta1 - theta2) + x2)/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: (-x1*np.cos(theta1 - theta2) + x2)/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'x2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-(x1*cos(theta1 - theta2) - x2)**2/(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2) + 1)/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: (-(x1*np.cos(theta1 - theta2) - x2)**2/(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) + 1)/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'x1*x2*sin(theta1 - theta2)/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: x1*x2*np.sin(theta1 - theta2)/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'x1*x2*(-x1*x2*sin(theta1 - theta2)**2/(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2) + cos(theta1 - theta2))/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: x1*x2*(-x1*x2*np.sin(theta1 - theta2)**2/(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) + np.cos(theta1 - theta2))/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-x1*x2*sin(theta1 - theta2)/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: -x1*x2*np.sin(theta1 - theta2)/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'x1*x2*(-x1*x2*sin(theta1 - theta2)**2/(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2) + cos(theta1 - theta2))/sqrt(x1**2 - 2*x1*x2*cos(theta1 - theta2) + x2**2)',
        'derivative_lambda': 'lambda args : (lambda x1,x2,theta1,theta2: x1*x2*(-x1*x2*np.sin(theta1 - theta2)**2/(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) + np.cos(theta1 - theta2))/np.sqrt(x1**2 - 2*x1*x2*np.cos(theta1 - theta2) + x2**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta2',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman30',
    'DescriptiveName': 'Feynman30, Lecture I.30.3',
    'Constraints': [
      {
        'name': 'Int_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sin(n*theta/2)**2/sin(theta/2)**2',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: np.sin(n*theta/2)**2/np.sin(theta/2)**2 )(*args)'
      },
      {
        'name': 'Int_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Int_0*n*sin(n*theta/2)*cos(n*theta/2)/sin(theta/2)**2 - Int_0*sin(n*theta/2)**2*cos(theta/2)/sin(theta/2)**3',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: Int_0*n*np.sin(n*theta/2)*np.cos(n*theta/2)/np.sin(theta/2)**2 - Int_0*np.sin(n*theta/2)**2*np.cos(theta/2)/np.sin(theta/2)**3 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'Int_0*(-n**2*(sin(n*theta/2)**2 - cos(n*theta/2)**2)/2 - 2*n*sin(n*theta/2)*cos(theta/2)*cos(n*theta/2)/sin(theta/2) + (1 + 3*cos(theta/2)**2/sin(theta/2)**2)*sin(n*theta/2)**2/2)/sin(theta/2)**2',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: Int_0*(-n**2*(np.sin(n*theta/2)**2 - np.cos(n*theta/2)**2)/2 - 2*n*np.sin(n*theta/2)*np.cos(theta/2)*np.cos(n*theta/2)/np.sin(theta/2) + (1 + 3*np.cos(theta/2)**2/np.sin(theta/2)**2)*np.sin(n*theta/2)**2/2)/np.sin(theta/2)**2 )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Int_0*theta*sin(n*theta/2)*cos(n*theta/2)/sin(theta/2)**2',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: Int_0*theta*np.sin(n*theta/2)*np.cos(n*theta/2)/np.sin(theta/2)**2 )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-Int_0*theta**2*(sin(n*theta/2)**2 - cos(n*theta/2)**2)/(2*sin(theta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda Int_0,theta,n: -Int_0*theta**2*(np.sin(n*theta/2)**2 - np.cos(n*theta/2)**2)/(2*np.sin(theta/2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Int_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman31',
    'DescriptiveName': 'Feynman31, Lecture I.30.5',
    'Constraints': [
      {
        'name': 'lambd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(d*n*sqrt(1 - lambd**2/(d**2*n**2)))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: 1/(d*n*np.sqrt(1 - lambd**2/(d**2*n**2))) )(*args)'
      },
      {
        'name': 'lambd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'lambd/(d**3*n**3*(1 - lambd**2/(d**2*n**2))**(3/2))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: lambd/(d**3*n**3*(1 - lambd**2/(d**2*n**2))**(3/2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-lambd/(d**2*n*sqrt(1 - lambd**2/(d**2*n**2)))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: -lambd/(d**2*n*np.sqrt(1 - lambd**2/(d**2*n**2))) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'lambd*(2 + lambd**2/(d**2*n**2*(1 - lambd**2/(d**2*n**2))))/(d**3*n*sqrt(1 - lambd**2/(d**2*n**2)))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: lambd*(2 + lambd**2/(d**2*n**2*(1 - lambd**2/(d**2*n**2))))/(d**3*n*np.sqrt(1 - lambd**2/(d**2*n**2))) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-lambd/(d*n**2*sqrt(1 - lambd**2/(d**2*n**2)))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: -lambd/(d*n**2*np.sqrt(1 - lambd**2/(d**2*n**2))) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'lambd*(2 + lambd**2/(d**2*n**2*(1 - lambd**2/(d**2*n**2))))/(d*n**3*sqrt(1 - lambd**2/(d**2*n**2)))',
        'derivative_lambda': 'lambda args : (lambda lambd,d,n: lambd*(2 + lambd**2/(d**2*n**2*(1 - lambd**2/(d**2*n**2))))/(d*n**3*np.sqrt(1 - lambd**2/(d**2*n**2))) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'lambd',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'd',
        'low': 2.0,
        'high': 5.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman32',
    'DescriptiveName': 'Feynman32, Lecture I.32.5',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'a**2*q/(3*pi*c**3*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: a**2*q/(3*np.pi*c**3*epsilon) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'a**2/(3*pi*c**3*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: a**2/(3*np.pi*c**3*epsilon) )(*args)'
      },
      {
        'name': 'a',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'a*q**2/(3*pi*c**3*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: a*q**2/(3*np.pi*c**3*epsilon) )(*args)'
      },
      {
        'name': 'a',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q**2/(3*pi*c**3*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: q**2/(3*np.pi*c**3*epsilon) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-a**2*q**2/(6*pi*c**3*epsilon**2)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: -a**2*q**2/(6*np.pi*c**3*epsilon**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'a**2*q**2/(3*pi*c**3*epsilon**3)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: a**2*q**2/(3*np.pi*c**3*epsilon**3) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-a**2*q**2/(2*pi*c**4*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: -a**2*q**2/(2*np.pi*c**4*epsilon) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*a**2*q**2/(pi*c**5*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,a,epsilon,c: 2*a**2*q**2/(np.pi*c**5*epsilon) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'a',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman33',
    'DescriptiveName': 'Feynman33, Lecture I.32.17',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*Ef**2*c*omega**4*r**2/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 4*np.pi*Ef**2*c*omega**4*r**2/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 0 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*Ef**2*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 4*np.pi*Ef**2*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*Ef*c*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 8*np.pi*Ef*c*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*c*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 8*np.pi*c*epsilon*omega**4*r**2/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*Ef**2*c*epsilon*omega**4*r/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 8*np.pi*Ef**2*c*epsilon*omega**4*r/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*Ef**2*c*epsilon*omega**4/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 8*np.pi*Ef**2*c*epsilon*omega**4/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-16*pi*Ef**2*c*epsilon*omega**5*r**2/(3*(omega**2 - omega_0**2)**3) + 16*pi*Ef**2*c*epsilon*omega**3*r**2/(3*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: -16*np.pi*Ef**2*c*epsilon*omega**5*r**2/(3*(omega**2 - omega_0**2)**3) + 16*np.pi*Ef**2*c*epsilon*omega**3*r**2/(3*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '16*pi*Ef**2*c*epsilon*omega**2*r**2*(omega**2*(6*omega**2/(omega**2 - omega_0**2) - 1)/(3*(omega**2 - omega_0**2)) - 8*omega**2/(3*(omega**2 - omega_0**2)) + 1)/(omega**2 - omega_0**2)**2',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 16*np.pi*Ef**2*c*epsilon*omega**2*r**2*(omega**2*(6*omega**2/(omega**2 - omega_0**2) - 1)/(3*(omega**2 - omega_0**2)) - 8*omega**2/(3*(omega**2 - omega_0**2)) + 1)/(omega**2 - omega_0**2)**2 )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '16*pi*Ef**2*c*epsilon*omega**4*omega_0*r**2/(3*(omega**2 - omega_0**2)**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 16*np.pi*Ef**2*c*epsilon*omega**4*omega_0*r**2/(3*(omega**2 - omega_0**2)**3) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '16*pi*Ef**2*c*epsilon*omega**4*r**2*(6*omega_0**2/(omega**2 - omega_0**2) + 1)/(3*(omega**2 - omega_0**2)**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef,r,omega,omega_0: 16*np.pi*Ef**2*c*epsilon*omega**4*r**2*(6*omega_0**2/(omega**2 - omega_0**2) + 1)/(3*(omega**2 - omega_0**2)**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'omega_0',
        'low': 3.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman34',
    'DescriptiveName': 'Feynman34, Lecture I.34.8',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*v/p',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: B*v/p )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*q/p',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: B*q/p )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*v/p',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: q*v/p )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: 0 )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-B*q*v/p**2',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: -B*q*v/p**2 )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*B*q*v/p**3',
        'derivative_lambda': 'lambda args : (lambda q,v,B,p: 2*B*q*v/p**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'p',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman35',
    'DescriptiveName': 'Feynman35, Lecture I.34.1',
    'Constraints': [
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-omega_0*v/(c**2*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: -omega_0*v/(c**2*(1 - v/c)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*omega_0*v*(1 + v/(c*(1 - v/c)))/(c**3*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: 2*omega_0*v*(1 + v/(c*(1 - v/c)))/(c**3*(1 - v/c)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega_0/(c*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: omega_0/(c*(1 - v/c)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*omega_0/(c**2*(1 - v/c)**3)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: 2*omega_0/(c**2*(1 - v/c)**3) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(1 - v/c)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: 1/(1 - v/c) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'omega_0',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman36',
    'DescriptiveName': 'Feynman36, Lecture I.34.14',
    'Constraints': [
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-omega_0*v/(c**2*sqrt(1 - v**2/c**2)) - omega_0*v**2*(1 + v/c)/(c**3*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: -omega_0*v/(c**2*np.sqrt(1 - v**2/c**2)) - omega_0*v**2*(1 + v/c)/(c**3*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega_0*v*(2 + 3*v*(1 + v/c)*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c*(1 - v**2/c**2)) + 2*v**2/(c**2*(1 - v**2/c**2)))/(c**3*sqrt(1 - v**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: omega_0*v*(2 + 3*v*(1 + v/c)*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c*(1 - v**2/c**2)) + 2*v**2/(c**2*(1 - v**2/c**2)))/(c**3*np.sqrt(1 - v**2/c**2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega_0/(c*sqrt(1 - v**2/c**2)) + omega_0*v*(1 + v/c)/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: omega_0/(c*np.sqrt(1 - v**2/c**2)) + omega_0*v*(1 + v/c)/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega_0*((1 + v/c)*(1 + 3*v**2/(c**2*(1 - v**2/c**2))) + 2*v/c)/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: omega_0*((1 + v/c)*(1 + 3*v**2/(c**2*(1 - v**2/c**2))) + 2*v/c)/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(1 + v/c)/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: (1 + v/c)/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda c,v,omega_0: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'omega_0',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman37',
    'DescriptiveName': 'Feynman37, Lecture I.34.27',
    'Constraints': [
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h/(2*pi)',
        'derivative_lambda': 'lambda args : (lambda omega,h: h/(2*np.pi) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda omega,h: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega/(2*pi)',
        'derivative_lambda': 'lambda args : (lambda omega,h: omega/(2*np.pi) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda omega,h: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman38',
    'DescriptiveName': 'Feynman38, Lecture I.37.4',
    'Constraints': [
      {
        'name': 'I1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '1 + sqrt(I1*I2)*cos(delta)/I1',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: 1 + np.sqrt(I1*I2)*np.cos(delta)/I1 )(*args)'
      },
      {
        'name': 'I1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-sqrt(I1*I2)*cos(delta)/(2*I1**2)',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: -np.sqrt(I1*I2)*np.cos(delta)/(2*I1**2) )(*args)'
      },
      {
        'name': 'I2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '1 + sqrt(I1*I2)*cos(delta)/I2',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: 1 + np.sqrt(I1*I2)*np.cos(delta)/I2 )(*args)'
      },
      {
        'name': 'I2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-sqrt(I1*I2)*cos(delta)/(2*I2**2)',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: -np.sqrt(I1*I2)*np.cos(delta)/(2*I2**2) )(*args)'
      },
      {
        'name': 'delta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-2*sqrt(I1*I2)*sin(delta)',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: -2*np.sqrt(I1*I2)*np.sin(delta) )(*args)'
      },
      {
        'name': 'delta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*sqrt(I1*I2)*cos(delta)',
        'derivative_lambda': 'lambda args : (lambda I1,I2,delta: -2*np.sqrt(I1*I2)*np.cos(delta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'I1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'I2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'delta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman39',
    'DescriptiveName': 'Feynman39, Lecture I.38.12',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-epsilon*h**2/(pi*m**2*q**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: -epsilon*h**2/(np.pi*m**2*q**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*epsilon*h**2/(pi*m**3*q**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: 2*epsilon*h**2/(np.pi*m**3*q**2) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*epsilon*h**2/(pi*m*q**3)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: -2*epsilon*h**2/(np.pi*m*q**3) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*epsilon*h**2/(pi*m*q**4)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: 6*epsilon*h**2/(np.pi*m*q**4) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*epsilon*h/(pi*m*q**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: 2*epsilon*h/(np.pi*m*q**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*epsilon/(pi*m*q**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: 2*epsilon/(np.pi*m*q**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2/(pi*m*q**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: h**2/(np.pi*m*q**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,q,h,epsilon: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman40',
    'DescriptiveName': 'Feynman40, Lecture I.39.1',
    'Constraints': [
      {
        'name': 'pr',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*V/2',
        'derivative_lambda': 'lambda args : (lambda pr,V: 3*V/2 )(*args)'
      },
      {
        'name': 'pr',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda pr,V: 0 )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pr/2',
        'derivative_lambda': 'lambda args : (lambda pr,V: 3*pr/2 )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda pr,V: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'pr',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'V',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman41',
    'DescriptiveName': 'Feynman41, Lecture I.39.11',
    'Constraints': [
      {
        'name': 'gamma',
        'order_derivative': 1,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: 0 )(*args)'
      },
      {
        'name': 'gamma',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: 0 )(*args)'
      },
      {
        'name': 'pr',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'V/(gamma - 1)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: V/(gamma - 1) )(*args)'
      },
      {
        'name': 'pr',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: 0 )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'pr/(gamma - 1)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: pr/(gamma - 1) )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,V: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'gamma',
        'low': 2.0,
        'high': 5.0
      },
      {
        'name': 'pr',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'V',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman42',
    'DescriptiveName': 'Feynman42, Lecture I.39.22',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*kb/V',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: T*kb/V )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: 0 )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'kb*n/V',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: kb*n/V )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: 0 )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-T*kb*n/V**2',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: -T*kb*n/V**2 )(*args)'
      },
      {
        'name': 'V',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*T*kb*n/V**3',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: 2*T*kb*n/V**3 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*n/V',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: T*n/V )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,T,V,kb: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'V',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman43',
    'DescriptiveName': 'Feynman43, Lecture I.40.1',
    'Constraints': [
      {
        'name': 'n_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'exp(-g*m*x/(T*kb))',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: np.exp(-g*m*x/(T*kb)) )(*args)'
      },
      {
        'name': 'n_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: 0 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-g*n_0*x*exp(-g*m*x/(T*kb))/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: -g*n_0*x*np.exp(-g*m*x/(T*kb))/(T*kb) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g**2*n_0*x**2*exp(-g*m*x/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: g**2*n_0*x**2*np.exp(-g*m*x/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-g*m*n_0*exp(-g*m*x/(T*kb))/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: -g*m*n_0*np.exp(-g*m*x/(T*kb))/(T*kb) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g**2*m**2*n_0*exp(-g*m*x/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: g**2*m**2*n_0*np.exp(-g*m*x/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g*m*n_0*x*exp(-g*m*x/(T*kb))/(T**2*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: g*m*n_0*x*np.exp(-g*m*x/(T*kb))/(T**2*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-g*m*n_0*x*(2 - g*m*x/(T*kb))*exp(-g*m*x/(T*kb))/(T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: -g*m*n_0*x*(2 - g*m*x/(T*kb))*np.exp(-g*m*x/(T*kb))/(T**3*kb) )(*args)'
      },
      {
        'name': 'g',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-m*n_0*x*exp(-g*m*x/(T*kb))/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: -m*n_0*x*np.exp(-g*m*x/(T*kb))/(T*kb) )(*args)'
      },
      {
        'name': 'g',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm**2*n_0*x**2*exp(-g*m*x/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: m**2*n_0*x**2*np.exp(-g*m*x/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g*m*n_0*x*exp(-g*m*x/(T*kb))/(T*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: g*m*n_0*x*np.exp(-g*m*x/(T*kb))/(T*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-g*m*n_0*x*(2 - g*m*x/(T*kb))*exp(-g*m*x/(T*kb))/(T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda n_0,m,x,T,g,kb: -g*m*n_0*x*(2 - g*m*x/(T*kb))*np.exp(-g*m*x/(T*kb))/(T*kb**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'g',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman44',
    'DescriptiveName': 'Feynman44, Lecture I.41.16',
    'Constraints': [
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '3*h*omega**2/(2*pi**3*c**2*(exp(h*omega/(2*pi*T*kb)) - 1)) - h**2*omega**3*exp(h*omega/(2*pi*T*kb))/(4*pi**4*T*c**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: 3*h*omega**2/(2*np.pi**3*c**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) - h**2*omega**3*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**4*T*c**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'h*omega*(3 - 3*h*omega*exp(h*omega/(2*pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)) - h**2*omega**2*(exp(h*omega/(2*pi*T*kb)) - 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(8*pi**2*T**2*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)))/(pi**3*c**2*(exp(h*omega/(2*pi*T*kb)) - 1))',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: h*omega*(3 - 3*h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) - h**2*omega**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(8*np.pi**2*T**2*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(np.pi**3*c**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*omega**4*exp(h*omega/(2*pi*T*kb))/(4*pi**4*T**2*c**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: h**2*omega**4*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**4*T**2*c**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-h**2*omega**4*(exp(h*omega/(2*pi*T*kb)) + h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) - h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(2*pi**4*T**3*c**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: -h**2*omega**4*(np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) - h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(2*np.pi**4*T**3*c**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'omega**3/(2*pi**3*c**2*(exp(h*omega/(2*pi*T*kb)) - 1)) - h*omega**4*exp(h*omega/(2*pi*T*kb))/(4*pi**4*T*c**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: omega**3/(2*np.pi**3*c**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) - h*omega**4*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**4*T*c**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-omega**4*(4*exp(h*omega/(2*pi*T*kb)) + h*omega*(exp(h*omega/(2*pi*T*kb)) - 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(pi*T*kb))/(8*pi**4*T*c**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: -omega**4*(4*np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*(np.exp(h*omega/(2*np.pi*T*kb)) - 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(np.pi*T*kb))/(8*np.pi**4*T*c**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*omega**4*exp(h*omega/(2*pi*T*kb))/(4*pi**4*T*c**2*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: h**2*omega**4*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**4*T*c**2*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-h**2*omega**4*(exp(h*omega/(2*pi*T*kb)) + h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) - h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(2*pi**4*T*c**2*kb**3*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: -h**2*omega**4*(np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) - h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(2*np.pi**4*T*c**2*kb**3*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-h*omega**3/(pi**3*c**3*(exp(h*omega/(2*pi*T*kb)) - 1))',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: -h*omega**3/(np.pi**3*c**3*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*h*omega**3/(pi**3*c**4*(exp(h*omega/(2*pi*T*kb)) - 1))',
        'derivative_lambda': 'lambda args : (lambda omega,T,h,kb,c: 3*h*omega**3/(np.pi**3*c**4*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman45',
    'DescriptiveName': 'Feynman45, Lecture I.43.16',
    'Constraints': [
      {
        'name': 'mu_drift',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Volt*q/d',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: Volt*q/d )(*args)'
      },
      {
        'name': 'mu_drift',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: 0 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Volt*mu_drift/d',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: Volt*mu_drift/d )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: 0 )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'mu_drift*q/d',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: mu_drift*q/d )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: 0 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Volt*mu_drift*q/d**2',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: -Volt*mu_drift*q/d**2 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Volt*mu_drift*q/d**3',
        'derivative_lambda': 'lambda args : (lambda mu_drift,q,Volt,d: 2*Volt*mu_drift*q/d**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mu_drift',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Volt',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman46',
    'DescriptiveName': 'Feynman46, Lecture I.43.31',
    'Constraints': [
      {
        'name': 'mob',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*kb',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: T*kb )(*args)'
      },
      {
        'name': 'mob',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: 0 )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'kb*mob',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: kb*mob )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*mob',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: T*mob )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mob,T,kb: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mob',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman47',
    'DescriptiveName': 'Feynman47, Lecture I.43.43',
    'Constraints': [
      {
        'name': 'gamma',
        'order_derivative': 1,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: 0 )(*args)'
      },
      {
        'name': 'gamma',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'v/(A*(gamma - 1))',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: v/(A*(gamma - 1)) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: 0 )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-kb*v/(A**2*(gamma - 1))',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: -kb*v/(A**2*(gamma - 1)) )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*kb*v/(A**3*(gamma - 1))',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: 2*kb*v/(A**3*(gamma - 1)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'kb/(A*(gamma - 1))',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: kb/(A*(gamma - 1)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,kb,A,v: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'gamma',
        'low': 2.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'A',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman48',
    'DescriptiveName': 'Feynman48, Lecture I.44.4',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'T*kb*log(V2/V1)',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: T*kb*log(V2/V1) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'T*n*log(V2/V1)',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: T*n*log(V2/V1) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: 0 )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'kb*n*log(V2/V1)',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: kb*n*log(V2/V1) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: 0 )(*args)'
      },
      {
        'name': 'V1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-T*kb*n/V1',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: -T*kb*n/V1 )(*args)'
      },
      {
        'name': 'V1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*kb*n/V1**2',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: T*kb*n/V1**2 )(*args)'
      },
      {
        'name': 'V2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'T*kb*n/V2',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: T*kb*n/V2 )(*args)'
      },
      {
        'name': 'V2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-T*kb*n/V2**2',
        'derivative_lambda': 'lambda args : (lambda n,kb,T,V1,V2: -T*kb*n/V2**2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'V1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'V2',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman49',
    'DescriptiveName': 'Feynman49, Lecture I.47.23',
    'Constraints': [
      {
        'name': 'gamma',
        'order_derivative': 1,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: 0 )(*args)'
      },
      {
        'name': 'gamma',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: 0 )(*args)'
      },
      {
        'name': 'pr',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(gamma*pr/rho)/(2*pr)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: np.sqrt(gamma*pr/rho)/(2*pr) )(*args)'
      },
      {
        'name': 'pr',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(gamma*pr/rho)/(4*pr**2)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: -np.sqrt(gamma*pr/rho)/(4*pr**2) )(*args)'
      },
      {
        'name': 'rho',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(gamma*pr/rho)/(2*rho)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: -np.sqrt(gamma*pr/rho)/(2*rho) )(*args)'
      },
      {
        'name': 'rho',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*sqrt(gamma*pr/rho)/(4*rho**2)',
        'derivative_lambda': 'lambda args : (lambda gamma,pr,rho: 3*np.sqrt(gamma*pr/rho)/(4*rho**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'gamma',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'pr',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'rho',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman50',
    'DescriptiveName': 'Feynman50, Lecture I.48.2',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**2/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda m,v,c: c**2/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,v,c: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*v/(1 - v**2/c**2)**(3/2)',
        'derivative_lambda': 'lambda args : (lambda m,v,c: m*v/(1 - v**2/c**2)**(3/2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(1 - v**2/c**2)**(3/2)',
        'derivative_lambda': 'lambda args : (lambda m,v,c: m*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(1 - v**2/c**2)**(3/2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*c*m/sqrt(1 - v**2/c**2) - m*v**2/(c*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m,v,c: 2*c*m/np.sqrt(1 - v**2/c**2) - m*v**2/(c*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*(2 + 3*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)) - 4*v**2/(c**2*(1 - v**2/c**2)))/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda m,v,c: m*(2 + 3*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)) - 4*v**2/(c**2*(1 - v**2/c**2)))/np.sqrt(1 - v**2/c**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman51',
    'DescriptiveName': 'Feynman51, Lecture I.50.26',
    'Constraints': [
      {
        'name': 'x1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'alpha*cos(omega*t)**2 + cos(omega*t)',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: alpha*np.cos(omega*t)**2 + np.cos(omega*t) )(*args)'
      },
      {
        'name': 'x1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: 0 )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'x1*(-2*alpha*t*sin(omega*t)*cos(omega*t) - t*sin(omega*t))',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: x1*(-2*alpha*t*np.sin(omega*t)*np.cos(omega*t) - t*np.sin(omega*t)) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-t**2*x1*(-2*alpha*sin(omega*t)**2 + 2*alpha*cos(omega*t)**2 + cos(omega*t))',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: -t**2*x1*(-2*alpha*np.sin(omega*t)**2 + 2*alpha*np.cos(omega*t)**2 + np.cos(omega*t)) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'x1*(-2*alpha*omega*sin(omega*t)*cos(omega*t) - omega*sin(omega*t))',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: x1*(-2*alpha*omega*np.sin(omega*t)*np.cos(omega*t) - omega*np.sin(omega*t)) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-omega**2*x1*(-2*alpha*sin(omega*t)**2 + 2*alpha*cos(omega*t)**2 + cos(omega*t))',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: -omega**2*x1*(-2*alpha*np.sin(omega*t)**2 + 2*alpha*np.cos(omega*t)**2 + np.cos(omega*t)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'x1*cos(omega*t)**2',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: x1*np.cos(omega*t)**2 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda x1,omega,t,alpha: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'x1',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 't',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman52',
    'DescriptiveName': 'Feynman52, Lecture II.2.42',
    'Constraints': [
      {
        'name': 'kappa',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'A*(-T1 + T2)/d',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: A*(-T1 + T2)/d )(*args)'
      },
      {
        'name': 'kappa',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: 0 )(*args)'
      },
      {
        'name': 'T1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-A*kappa/d',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: -A*kappa/d )(*args)'
      },
      {
        'name': 'T1',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: 0 )(*args)'
      },
      {
        'name': 'T2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'A*kappa/d',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: A*kappa/d )(*args)'
      },
      {
        'name': 'T2',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: 0 )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'kappa*(-T1 + T2)/d',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: kappa*(-T1 + T2)/d )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: 0 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-A*kappa*(-T1 + T2)/d**2',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: -A*kappa*(-T1 + T2)/d**2 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*A*kappa*(T1 - T2)/d**3',
        'derivative_lambda': 'lambda args : (lambda kappa,T1,T2,A,d: -2*A*kappa*(T1 - T2)/d**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'kappa',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'A',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman53',
    'DescriptiveName': 'Feynman53, Lecture II.3.24',
    'Constraints': [
      {
        'name': 'Pwr',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi*r**2)',
        'derivative_lambda': 'lambda args : (lambda Pwr,r: 1/(4*np.pi*r**2) )(*args)'
      },
      {
        'name': 'Pwr',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Pwr,r: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Pwr/(2*pi*r**3)',
        'derivative_lambda': 'lambda args : (lambda Pwr,r: -Pwr/(2*np.pi*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*Pwr/(2*pi*r**4)',
        'derivative_lambda': 'lambda args : (lambda Pwr,r: 3*Pwr/(2*np.pi*r**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Pwr',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman54',
    'DescriptiveName': 'Feynman54, Lecture II.4.23',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi*epsilon*r)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: 1/(4*np.pi*epsilon*r) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/(4*pi*epsilon**2*r)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: -q/(4*np.pi*epsilon**2*r) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*epsilon**3*r)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: q/(2*np.pi*epsilon**3*r) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: -q/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r: q/(2*np.pi*epsilon*r**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman55',
    'DescriptiveName': 'Feynman55, Lecture II.6.11',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-p_d*cos(theta)/(4*pi*epsilon**2*r**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -p_d*np.cos(theta)/(4*np.pi*epsilon**2*r**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'p_d*cos(theta)/(2*pi*epsilon**3*r**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: p_d*np.cos(theta)/(2*np.pi*epsilon**3*r**2) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'cos(theta)/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: np.cos(theta)/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-p_d*sin(theta)/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -p_d*np.sin(theta)/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-p_d*cos(theta)/(4*pi*epsilon*r**2)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -p_d*np.cos(theta)/(4*np.pi*epsilon*r**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-p_d*cos(theta)/(2*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -p_d*np.cos(theta)/(2*np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '3*p_d*cos(theta)/(2*pi*epsilon*r**4)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 3*p_d*np.cos(theta)/(2*np.pi*epsilon*r**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman56',
    'DescriptiveName': 'Feynman56, Lecture II.6.15a',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*p_d*z*sqrt(x**2 + y**2)/(4*pi*epsilon**2*r**5)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: -3*p_d*z*np.sqrt(x**2 + y**2)/(4*np.pi*epsilon**2*r**5) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*p_d*z*sqrt(x**2 + y**2)/(2*pi*epsilon**3*r**5)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 3*p_d*z*np.sqrt(x**2 + y**2)/(2*np.pi*epsilon**3*r**5) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*z*sqrt(x**2 + y**2)/(4*pi*epsilon*r**5)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 3*z*np.sqrt(x**2 + y**2)/(4*np.pi*epsilon*r**5) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-15*p_d*z*sqrt(x**2 + y**2)/(4*pi*epsilon*r**6)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: -15*p_d*z*np.sqrt(x**2 + y**2)/(4*np.pi*epsilon*r**6) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '45*p_d*z*sqrt(x**2 + y**2)/(2*pi*epsilon*r**7)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 45*p_d*z*np.sqrt(x**2 + y**2)/(2*np.pi*epsilon*r**7) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*p_d*x*z/(4*pi*epsilon*r**5*sqrt(x**2 + y**2))',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 3*p_d*x*z/(4*np.pi*epsilon*r**5*np.sqrt(x**2 + y**2)) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-3*p_d*z*(x**2/(x**2 + y**2) - 1)/(4*pi*epsilon*r**5*sqrt(x**2 + y**2))',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: -3*p_d*z*(x**2/(x**2 + y**2) - 1)/(4*np.pi*epsilon*r**5*np.sqrt(x**2 + y**2)) )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*p_d*y*z/(4*pi*epsilon*r**5*sqrt(x**2 + y**2))',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 3*p_d*y*z/(4*np.pi*epsilon*r**5*np.sqrt(x**2 + y**2)) )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-3*p_d*z*(y**2/(x**2 + y**2) - 1)/(4*pi*epsilon*r**5*sqrt(x**2 + y**2))',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: -3*p_d*z*(y**2/(x**2 + y**2) - 1)/(4*np.pi*epsilon*r**5*np.sqrt(x**2 + y**2)) )(*args)'
      },
      {
        'name': 'z',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*p_d*sqrt(x**2 + y**2)/(4*pi*epsilon*r**5)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 3*p_d*np.sqrt(x**2 + y**2)/(4*np.pi*epsilon*r**5) )(*args)'
      },
      {
        'name': 'z',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,r,x,y,z: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'y',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'z',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman57',
    'DescriptiveName': 'Feynman57, Lecture II.6.15b',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-3*p_d*sin(theta)*cos(theta)/(4*pi*epsilon**2*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -3*p_d*np.sin(theta)*np.cos(theta)/(4*np.pi*epsilon**2*r**3) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '3*p_d*sin(theta)*cos(theta)/(2*pi*epsilon**3*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 3*p_d*np.sin(theta)*np.cos(theta)/(2*np.pi*epsilon**3*r**3) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '3*sin(theta)*cos(theta)/(4*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 3*np.sin(theta)*np.cos(theta)/(4*np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-3*p_d*sin(theta)**2/(4*pi*epsilon*r**3) + 3*p_d*cos(theta)**2/(4*pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -3*p_d*np.sin(theta)**2/(4*np.pi*epsilon*r**3) + 3*p_d*np.cos(theta)**2/(4*np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-3*p_d*sin(theta)*cos(theta)/(pi*epsilon*r**3)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -3*p_d*np.sin(theta)*np.cos(theta)/(np.pi*epsilon*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-9*p_d*sin(theta)*cos(theta)/(4*pi*epsilon*r**4)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: -9*p_d*np.sin(theta)*np.cos(theta)/(4*np.pi*epsilon*r**4) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '9*p_d*sin(theta)*cos(theta)/(pi*epsilon*r**5)',
        'derivative_lambda': 'lambda args : (lambda epsilon,p_d,theta,r: 9*p_d*np.sin(theta)*np.cos(theta)/(np.pi*epsilon*r**5) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman58',
    'DescriptiveName': 'Feynman58, Lecture II.8.7',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*q/(10*pi*d*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: 3*q/(10*np.pi*d*epsilon) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3/(10*pi*d*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: 3/(10*np.pi*d*epsilon) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*q**2/(20*pi*d*epsilon**2)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: -3*q**2/(20*np.pi*d*epsilon**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*q**2/(10*pi*d*epsilon**3)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: 3*q**2/(10*np.pi*d*epsilon**3) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*q**2/(20*pi*d**2*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: -3*q**2/(20*np.pi*d**2*epsilon) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*q**2/(10*pi*d**3*epsilon)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,d: 3*q**2/(10*np.pi*d**3*epsilon) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman59',
    'DescriptiveName': 'Feynman59, Lecture II.8.31',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef**2/2',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: Ef**2/2 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: Ef*epsilon )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: epsilon )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman60',
    'DescriptiveName': 'Feynman60, Lecture II.10.9',
    'Constraints': [
      {
        'name': 'sigma_den',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(epsilon*(chi + 1))',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: 1/(epsilon*(chi + 1)) )(*args)'
      },
      {
        'name': 'sigma_den',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sigma_den/(epsilon**2*(chi + 1))',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: -sigma_den/(epsilon**2*(chi + 1)) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*sigma_den/(epsilon**3*(chi + 1))',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: 2*sigma_den/(epsilon**3*(chi + 1)) )(*args)'
      },
      {
        'name': 'chi',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sigma_den/(epsilon*(chi + 1)**2)',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: -sigma_den/(epsilon*(chi + 1)**2) )(*args)'
      },
      {
        'name': 'chi',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*sigma_den/(epsilon*(chi + 1)**3)',
        'derivative_lambda': 'lambda args : (lambda sigma_den,epsilon,chi: 2*sigma_den/(epsilon*(chi + 1)**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'sigma_den',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'chi',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman61',
    'DescriptiveName': 'Feynman61, Lecture II.11.3',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef/(m*(-omega**2 + omega_0**2))',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: Ef/(m*(-omega**2 + omega_0**2)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(m*(-omega**2 + omega_0**2))',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: q/(m*(-omega**2 + omega_0**2)) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: 0 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Ef*q/(m**2*(-omega**2 + omega_0**2))',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: -Ef*q/(m**2*(-omega**2 + omega_0**2)) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-2*Ef*q/(m**3*(omega**2 - omega_0**2))',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: -2*Ef*q/(m**3*(omega**2 - omega_0**2)) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*Ef*omega_0*q/(m*(-omega**2 + omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: -2*Ef*omega_0*q/(m*(-omega**2 + omega_0**2)**2) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-2*Ef*q*(4*omega_0**2/(omega**2 - omega_0**2) + 1)/(m*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: -2*Ef*q*(4*omega_0**2/(omega**2 - omega_0**2) + 1)/(m*(omega**2 - omega_0**2)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*omega*q/(m*(-omega**2 + omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: 2*Ef*omega*q/(m*(-omega**2 + omega_0**2)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-2*Ef*q*(4*omega**2/(omega**2 - omega_0**2) - 1)/(m*(omega**2 - omega_0**2)**2)',
        'derivative_lambda': 'lambda args : (lambda q,Ef,m,omega_0,omega: -2*Ef*q*(4*omega**2/(omega**2 - omega_0**2) - 1)/(m*(omega**2 - omega_0**2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega_0',
        'low': 3.0,
        'high': 5.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Feynman62',
    'DescriptiveName': 'Feynman62, Lecture II.11.17',
    'Constraints': [
      {
        'name': 'n_0',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Ef*p_d*cos(theta)/(T*kb) + 1',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: Ef*p_d*np.cos(theta)/(T*kb) + 1 )(*args)'
      },
      {
        'name': 'n_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-Ef*n_0*p_d*cos(theta)/(T*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: -Ef*n_0*p_d*np.cos(theta)/(T*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*Ef*n_0*p_d*cos(theta)/(T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: 2*Ef*n_0*p_d*np.cos(theta)/(T*kb**3) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-Ef*n_0*p_d*cos(theta)/(T**2*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: -Ef*n_0*p_d*np.cos(theta)/(T**2*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*Ef*n_0*p_d*cos(theta)/(T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: 2*Ef*n_0*p_d*np.cos(theta)/(T**3*kb) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Ef*n_0*p_d*sin(theta)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: -Ef*n_0*p_d*np.sin(theta)/(T*kb) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-Ef*n_0*p_d*cos(theta)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: -Ef*n_0*p_d*np.cos(theta)/(T*kb) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Ef*n_0*cos(theta)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: Ef*n_0*np.cos(theta)/(T*kb) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'n_0*p_d*cos(theta)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: n_0*p_d*np.cos(theta)/(T*kb) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,theta,p_d,Ef: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n_0',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman63',
    'DescriptiveName': 'Feynman63, Lecture II.11.20',
    'Constraints': [
      {
        'name': 'n_rho',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef*p_d**2/(3*T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: Ef*p_d**2/(3*T*kb) )(*args)'
      },
      {
        'name': 'n_rho',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 0 )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*n_rho*p_d/(3*T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 2*Ef*n_rho*p_d/(3*T*kb) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*n_rho/(3*T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 2*Ef*n_rho/(3*T*kb) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'n_rho*p_d**2/(3*T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: n_rho*p_d**2/(3*T*kb) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Ef*n_rho*p_d**2/(3*T*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: -Ef*n_rho*p_d**2/(3*T*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*n_rho*p_d**2/(3*T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 2*Ef*n_rho*p_d**2/(3*T*kb**3) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Ef*n_rho*p_d**2/(3*T**2*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: -Ef*n_rho*p_d**2/(3*T**2*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*n_rho*p_d**2/(3*T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,p_d,Ef,kb,T: 2*Ef*n_rho*p_d**2/(3*T**3*kb) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n_rho',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman64',
    'DescriptiveName': 'Feynman64, Lecture II.11.27',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef*alpha**2*epsilon*n/(3*(-alpha*n/3 + 1)**2) + Ef*alpha*epsilon/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: Ef*alpha**2*epsilon*n/(3*(-alpha*n/3 + 1)**2) + Ef*alpha*epsilon/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*Ef*alpha**2*epsilon*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: 6*Ef*alpha**2*epsilon*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef*alpha*epsilon*n**2/(3*(-alpha*n/3 + 1)**2) + Ef*epsilon*n/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: Ef*alpha*epsilon*n**2/(3*(-alpha*n/3 + 1)**2) + Ef*epsilon*n/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*Ef*epsilon*n**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: 6*Ef*epsilon*n**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef*alpha*n/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: Ef*alpha*n/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*epsilon*n/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: alpha*epsilon*n/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,alpha,epsilon,Ef: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 0.0,
        'high': 1.0
      },
      {
        'name': 'alpha',
        'low': 0.0,
        'high': 1.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Feynman65',
    'DescriptiveName': 'Feynman65, Lecture II.11.28',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha**2*n/(3*(-alpha*n/3 + 1)**2) + alpha/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha: alpha**2*n/(3*(-alpha*n/3 + 1)**2) + alpha/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*alpha**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2',
        'derivative_lambda': 'lambda args : (lambda n,alpha: 6*alpha**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*n**2/(3*(-alpha*n/3 + 1)**2) + n/(-alpha*n/3 + 1)',
        'derivative_lambda': 'lambda args : (lambda n,alpha: alpha*n**2/(3*(-alpha*n/3 + 1)**2) + n/(-alpha*n/3 + 1) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*n**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2',
        'derivative_lambda': 'lambda args : (lambda n,alpha: 6*n**2*(-alpha*n/(alpha*n - 3) + 1)/(alpha*n - 3)**2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 0.0,
        'high': 1.0
      },
      {
        'name': 'alpha',
        'low': 0.0,
        'high': 1.0
      }
    ]
  },
  {
    'EquationName': 'Feynman67',
    'DescriptiveName': 'Feynman67, Lecture II.13.23',
    'Constraints': [
      {
        'name': 'rho_c_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: 1/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'rho_c_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'rho_c_0*v/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0*v/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'rho_c_0*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0*(1 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-rho_c_0*v**2/(c**3*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: -rho_c_0*v**2/(c**3*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*rho_c_0*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: 3*rho_c_0*v**2*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'rho_c_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman68',
    'DescriptiveName': 'Feynman68, Lecture II.13.34',
    'Constraints': [
      {
        'name': 'rho_c_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'v/sqrt(1 - v**2/c**2)',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: v/np.sqrt(1 - v**2/c**2) )(*args)'
      },
      {
        'name': 'rho_c_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'rho_c_0/sqrt(1 - v**2/c**2) + rho_c_0*v**2/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0/np.sqrt(1 - v**2/c**2) + rho_c_0*v**2/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'rho_c_0*v*(3 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: rho_c_0*v*(3 + 3*v**2/(c**2*(1 - v**2/c**2)))/(c**2*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-rho_c_0*v**3/(c**3*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: -rho_c_0*v**3/(c**3*(1 - v**2/c**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*rho_c_0*v**3*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,v,c: 3*rho_c_0*v**3*(1 + v**2/(c**2*(1 - v**2/c**2)))/(c**4*(1 - v**2/c**2)**(3/2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'rho_c_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman69',
    'DescriptiveName': 'Feynman69, Lecture II.15.4',
    'Constraints': [
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-B*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: -B*np.cos(theta) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-mom*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: -mom*np.cos(theta) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'B*mom*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: B*mom*np.sin(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'B*mom*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda mom,B,theta: B*mom*np.cos(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman70',
    'DescriptiveName': 'Feynman70, Lecture II.15.5',
    'Constraints': [
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-Ef*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: -Ef*np.cos(theta) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-p_d*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: -p_d*np.cos(theta) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Ef*p_d*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: Ef*p_d*np.sin(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'Ef*p_d*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,theta: Ef*p_d*np.cos(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman71',
    'DescriptiveName': 'Feynman71, Lecture II.21.32',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi*epsilon*r*(1 - v/c))',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: 1/(4*np.pi*epsilon*r*(1 - v/c)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/(4*pi*epsilon**2*r*(1 - v/c))',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: -q/(4*np.pi*epsilon**2*r*(1 - v/c)) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*epsilon**3*r*(1 - v/c))',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: q/(2*np.pi*epsilon**3*r*(1 - v/c)) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/(4*pi*epsilon*r**2*(1 - v/c))',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: -q/(4*np.pi*epsilon*r**2*(1 - v/c)) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*epsilon*r**3*(1 - v/c))',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: q/(2*np.pi*epsilon*r**3*(1 - v/c)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(4*pi*c*epsilon*r*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: q/(4*np.pi*c*epsilon*r*(1 - v/c)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*c**2*epsilon*r*(1 - v/c)**3)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: q/(2*np.pi*c**2*epsilon*r*(1 - v/c)**3) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q*v/(4*pi*c**2*epsilon*r*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: -q*v/(4*np.pi*c**2*epsilon*r*(1 - v/c)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*v*(1 + v/(c*(1 - v/c)))/(2*pi*c**3*epsilon*r*(1 - v/c)**2)',
        'derivative_lambda': 'lambda args : (lambda q,epsilon,r,v,c: q*v*(1 + v/(c*(1 - v/c)))/(2*np.pi*c**3*epsilon*r*(1 - v/c)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 3.0,
        'high': 10.0
      }
    ]
  },
  {
    'EquationName': 'Feynman72',
    'DescriptiveName': 'Feynman72, Lecture II.24.17',
    'Constraints': [
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega/(c**2*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: omega/(c**2*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '(1 - omega**2/(c**2*(-pi**2/d**2 + omega**2/c**2)))/(c**2*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: (1 - omega**2/(c**2*(-np.pi**2/d**2 + omega**2/c**2)))/(c**2*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-omega**2/(c**3*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: -omega**2/(c**3*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega**2*(3 - omega**2/(c**2*(-pi**2/d**2 + omega**2/c**2)))/(c**4*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: omega**2*(3 - omega**2/(c**2*(-np.pi**2/d**2 + omega**2/c**2)))/(c**4*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'pi**2/(d**3*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: np.pi**2/(d**3*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-pi**2*(3 + pi**2/(d**2*(-pi**2/d**2 + omega**2/c**2)))/(d**4*sqrt(-pi**2/d**2 + omega**2/c**2))',
        'derivative_lambda': 'lambda args : (lambda omega,c,d: -np.pi**2*(3 + np.pi**2/(d**2*(-np.pi**2/d**2 + omega**2/c**2)))/(d**4*np.sqrt(-np.pi**2/d**2 + omega**2/c**2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'omega',
        'low': 4.0,
        'high': 6.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'd',
        'low': 2.0,
        'high': 4.0
      }
    ]
  },
  {
    'EquationName': 'Feynman73',
    'DescriptiveName': 'Feynman73, Lecture II.27.16',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef**2*c',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: Ef**2*c )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: 0 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef**2*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: Ef**2*epsilon )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*c*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: 2*Ef*c*epsilon )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*c*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,c,Ef: 2*c*epsilon )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman74',
    'DescriptiveName': 'Feynman74, Lecture II.27.18',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Ef**2',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: Ef**2 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*Ef*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: 2*Ef*epsilon )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*epsilon',
        'derivative_lambda': 'lambda args : (lambda epsilon,Ef: 2*epsilon )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman75',
    'DescriptiveName': 'Feynman75, Lecture II.34.2a',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'v/(2*pi*r)',
        'derivative_lambda': 'lambda args : (lambda q,v,r: v/(2*np.pi*r) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,r: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*r)',
        'derivative_lambda': 'lambda args : (lambda q,v,r: q/(2*np.pi*r) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,r: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q*v/(2*pi*r**2)',
        'derivative_lambda': 'lambda args : (lambda q,v,r: -q*v/(2*np.pi*r**2) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*v/(pi*r**3)',
        'derivative_lambda': 'lambda args : (lambda q,v,r: q*v/(np.pi*r**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman76',
    'DescriptiveName': 'Feynman76, Lecture II.34.2',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'r*v/2',
        'derivative_lambda': 'lambda args : (lambda q,v,r: r*v/2 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,r: 0 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*r/2',
        'derivative_lambda': 'lambda args : (lambda q,v,r: q*r/2 )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,r: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*v/2',
        'derivative_lambda': 'lambda args : (lambda q,v,r: q*v/2 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,v,r: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman77',
    'DescriptiveName': 'Feynman77, Lecture II.34.11',
    'Constraints': [
      {
        'name': 'g_',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*q/(2*m)',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: B*q/(2*m) )(*args)'
      },
      {
        'name': 'g_',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: 0 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*g_/(2*m)',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: B*g_/(2*m) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'g_*q/(2*m)',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: g_*q/(2*m) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: 0 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-B*g_*q/(2*m**2)',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: -B*g_*q/(2*m**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*g_*q/m**3',
        'derivative_lambda': 'lambda args : (lambda g_,q,B,m: B*g_*q/m**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'g_',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman78',
    'DescriptiveName': 'Feynman78, Lecture II.34.29a',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h/(4*pi*m)',
        'derivative_lambda': 'lambda args : (lambda q,h,m: h/(4*np.pi*m) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,h,m: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(4*pi*m)',
        'derivative_lambda': 'lambda args : (lambda q,h,m: q/(4*np.pi*m) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,h,m: 0 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-h*q/(4*pi*m**2)',
        'derivative_lambda': 'lambda args : (lambda q,h,m: -h*q/(4*np.pi*m**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h*q/(2*pi*m**3)',
        'derivative_lambda': 'lambda args : (lambda q,h,m: h*q/(2*np.pi*m**3) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman79',
    'DescriptiveName': 'Feynman79, Lecture II.34.29b',
    'Constraints': [
      {
        'name': 'g_',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*pi*B*Jz*mom/h',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 2*np.pi*B*Jz*mom/h )(*args)'
      },
      {
        'name': 'g_',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*pi*B*Jz*g_*mom/h**2',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: -2*np.pi*B*Jz*g_*mom/h**2 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*B*Jz*g_*mom/h**3',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 4*np.pi*B*Jz*g_*mom/h**3 )(*args)'
      },
      {
        'name': 'Jz',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*pi*B*g_*mom/h',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 2*np.pi*B*g_*mom/h )(*args)'
      },
      {
        'name': 'Jz',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 0 )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*pi*B*Jz*g_/h',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 2*np.pi*B*Jz*g_/h )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*pi*Jz*g_*mom/h',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 2*np.pi*Jz*g_*mom/h )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda g_,h,Jz,mom,B: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'g_',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Jz',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman80',
    'DescriptiveName': 'Feynman80, Lecture II.35.18',
    'Constraints': [
      {
        'name': 'n_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: 1/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb))) )(*args)'
      },
      {
        'name': 'n_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'n_0*(B*mom*exp(B*mom/(T*kb))/(T*kb**2) - B*mom*exp(-B*mom/(T*kb))/(T*kb**2))/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: n_0*(B*mom*np.exp(B*mom/(T*kb))/(T*kb**2) - B*mom*np.exp(-B*mom/(T*kb))/(T*kb**2))/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-B*mom*n_0*(-2*B*mom*(exp(B*mom/(T*kb)) - exp(-B*mom/(T*kb)))**2/(T*kb*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))) + B*mom*exp(B*mom/(T*kb))/(T*kb) + B*mom*exp(-B*mom/(T*kb))/(T*kb) + 2*exp(B*mom/(T*kb)) - 2*exp(-B*mom/(T*kb)))/(T*kb**3*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: -B*mom*n_0*(-2*B*mom*(np.exp(B*mom/(T*kb)) - np.exp(-B*mom/(T*kb)))**2/(T*kb*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))) + B*mom*np.exp(B*mom/(T*kb))/(T*kb) + B*mom*np.exp(-B*mom/(T*kb))/(T*kb) + 2*np.exp(B*mom/(T*kb)) - 2*np.exp(-B*mom/(T*kb)))/(T*kb**3*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'n_0*(B*mom*exp(B*mom/(T*kb))/(T**2*kb) - B*mom*exp(-B*mom/(T*kb))/(T**2*kb))/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: n_0*(B*mom*np.exp(B*mom/(T*kb))/(T**2*kb) - B*mom*np.exp(-B*mom/(T*kb))/(T**2*kb))/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-B*mom*n_0*(-2*B*mom*(exp(B*mom/(T*kb)) - exp(-B*mom/(T*kb)))**2/(T*kb*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))) + B*mom*exp(B*mom/(T*kb))/(T*kb) + B*mom*exp(-B*mom/(T*kb))/(T*kb) + 2*exp(B*mom/(T*kb)) - 2*exp(-B*mom/(T*kb)))/(T**3*kb*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2)',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: -B*mom*n_0*(-2*B*mom*(np.exp(B*mom/(T*kb)) - np.exp(-B*mom/(T*kb)))**2/(T*kb*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))) + B*mom*np.exp(B*mom/(T*kb))/(T*kb) + B*mom*np.exp(-B*mom/(T*kb))/(T*kb) + 2*np.exp(B*mom/(T*kb)) - 2*np.exp(-B*mom/(T*kb)))/(T**3*kb*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'n_0*(-B*exp(B*mom/(T*kb))/(T*kb) + B*exp(-B*mom/(T*kb))/(T*kb))/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: n_0*(-B*np.exp(B*mom/(T*kb))/(T*kb) + B*np.exp(-B*mom/(T*kb))/(T*kb))/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'B**2*n_0*(2*(exp(B*mom/(T*kb)) - exp(-B*mom/(T*kb)))**2/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2 - 1)/(T**2*kb**2*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb))))',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: B**2*n_0*(2*(np.exp(B*mom/(T*kb)) - np.exp(-B*mom/(T*kb)))**2/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 - 1)/(T**2*kb**2*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'n_0*(-mom*exp(B*mom/(T*kb))/(T*kb) + mom*exp(-B*mom/(T*kb))/(T*kb))/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: n_0*(-mom*np.exp(B*mom/(T*kb))/(T*kb) + mom*np.exp(-B*mom/(T*kb))/(T*kb))/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'mom**2*n_0*(2*(exp(B*mom/(T*kb)) - exp(-B*mom/(T*kb)))**2/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))**2 - 1)/(T**2*kb**2*(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb))))',
        'derivative_lambda': 'lambda args : (lambda n_0,kb,T,mom,B: mom**2*n_0*(2*(np.exp(B*mom/(T*kb)) - np.exp(-B*mom/(T*kb)))**2/(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))**2 - 1)/(T**2*kb**2*(np.exp(B*mom/(T*kb)) + np.exp(-B*mom/(T*kb)))) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n_0',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'mom',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman81',
    'DescriptiveName': 'Feynman81, Lecture II.35.21',
    'Constraints': [
      {
        'name': 'n_rho',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'mom*tanh(B*mom/(T*kb))',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: mom*np.tanh(B*mom/(T*kb)) )(*args)'
      },
      {
        'name': 'n_rho',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: 0 )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*mom*n_rho*(1 - tanh(B*mom/(T*kb))**2)/(T*kb) + n_rho*tanh(B*mom/(T*kb))',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: B*mom*n_rho*(1 - np.tanh(B*mom/(T*kb))**2)/(T*kb) + n_rho*np.tanh(B*mom/(T*kb)) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*B*n_rho*(B*mom*tanh(B*mom/(T*kb))/(T*kb) - 1)*(tanh(B*mom/(T*kb))**2 - 1)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: 2*B*n_rho*(B*mom*np.tanh(B*mom/(T*kb))/(T*kb) - 1)*(np.tanh(B*mom/(T*kb))**2 - 1)/(T*kb) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'monotonic increasing',
        'derivative': 'mom**2*n_rho*(1 - tanh(B*mom/(T*kb))**2)/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: mom**2*n_rho*(1 - np.tanh(B*mom/(T*kb))**2)/(T*kb) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'monotonic decreasing',
        'derivative': '2*mom**3*n_rho*(tanh(B*mom/(T*kb))**2 - 1)*tanh(B*mom/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: 2*mom**3*n_rho*(np.tanh(B*mom/(T*kb))**2 - 1)*np.tanh(B*mom/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'monotonic decreasing',
        'derivative': '-B*mom**2*n_rho*(1 - tanh(B*mom/(T*kb))**2)/(T*kb**2)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: -B*mom**2*n_rho*(1 - np.tanh(B*mom/(T*kb))**2)/(T*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*B*mom**2*n_rho*(B*mom*tanh(B*mom/(T*kb))/(T*kb) - 1)*(tanh(B*mom/(T*kb))**2 - 1)/(T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: 2*B*mom**2*n_rho*(B*mom*np.tanh(B*mom/(T*kb))/(T*kb) - 1)*(np.tanh(B*mom/(T*kb))**2 - 1)/(T*kb**3) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'monotonic decreasing',
        'derivative': '-B*mom**2*n_rho*(1 - tanh(B*mom/(T*kb))**2)/(T**2*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: -B*mom**2*n_rho*(1 - np.tanh(B*mom/(T*kb))**2)/(T**2*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*B*mom**2*n_rho*(B*mom*tanh(B*mom/(T*kb))/(T*kb) - 1)*(tanh(B*mom/(T*kb))**2 - 1)/(T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda n_rho,mom,B,kb,T: 2*B*mom**2*n_rho*(B*mom*np.tanh(B*mom/(T*kb))/(T*kb) - 1)*(np.tanh(B*mom/(T*kb))**2 - 1)/(T**3*kb) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n_rho',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman82',
    'DescriptiveName': 'Feynman82, Lecture II.36.38',
    'Constraints': [
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'H/(T*kb) + M*alpha/(T*c**2*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: H/(T*kb) + M*alpha/(T*c**2*epsilon*kb) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 0 )(*args)'
      },
      {
        'name': 'H',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'mom/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: mom/(T*kb) )(*args)'
      },
      {
        'name': 'H',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 0 )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-H*mom/(T*kb**2) - M*alpha*mom/(T*c**2*epsilon*kb**2)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: -H*mom/(T*kb**2) - M*alpha*mom/(T*c**2*epsilon*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*mom*(H + M*alpha/(c**2*epsilon))/(T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 2*mom*(H + M*alpha/(c**2*epsilon))/(T*kb**3) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-H*mom/(T**2*kb) - M*alpha*mom/(T**2*c**2*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: -H*mom/(T**2*kb) - M*alpha*mom/(T**2*c**2*epsilon*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*mom*(H + M*alpha/(c**2*epsilon))/(T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 2*mom*(H + M*alpha/(c**2*epsilon))/(T**3*kb) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'M*mom/(T*c**2*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: M*mom/(T*c**2*epsilon*kb) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 0 )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-M*alpha*mom/(T*c**2*epsilon**2*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: -M*alpha*mom/(T*c**2*epsilon**2*kb) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*M*alpha*mom/(T*c**2*epsilon**3*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 2*M*alpha*mom/(T*c**2*epsilon**3*kb) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*M*alpha*mom/(T*c**3*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: -2*M*alpha*mom/(T*c**3*epsilon*kb) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '6*M*alpha*mom/(T*c**4*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 6*M*alpha*mom/(T*c**4*epsilon*kb) )(*args)'
      },
      {
        'name': 'M',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*mom/(T*c**2*epsilon*kb)',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: alpha*mom/(T*c**2*epsilon*kb) )(*args)'
      },
      {
        'name': 'M',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mom',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'H',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'M',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Feynman83',
    'DescriptiveName': 'Feynman83, Lecture II.37.1',
    'Constraints': [
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*(chi + 1)',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: B*(chi + 1) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'mom*(chi + 1)',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: mom*(chi + 1) )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: 0 )(*args)'
      },
      {
        'name': 'chi',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'B*mom',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: B*mom )(*args)'
      },
      {
        'name': 'chi',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,chi: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'chi',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman84',
    'DescriptiveName': 'Feynman84, Lecture II.38.3',
    'Constraints': [
      {
        'name': 'Y',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'A*x/d',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: A*x/d )(*args)'
      },
      {
        'name': 'Y',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: 0 )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Y*x/d',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: Y*x/d )(*args)'
      },
      {
        'name': 'A',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: 0 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-A*Y*x/d**2',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: -A*Y*x/d**2 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*A*Y*x/d**3',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: 2*A*Y*x/d**3 )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'A*Y/d',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: A*Y/d )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Y,A,d,x: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Y',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'A',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman85',
    'DescriptiveName': 'Feynman85, Lecture II.38.14',
    'Constraints': [
      {
        'name': 'Y',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(2*sigma + 2)',
        'derivative_lambda': 'lambda args : (lambda Y,sigma: 1/(2*sigma + 2) )(*args)'
      },
      {
        'name': 'Y',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Y,sigma: 0 )(*args)'
      },
      {
        'name': 'sigma',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*Y/(2*sigma + 2)**2',
        'derivative_lambda': 'lambda args : (lambda Y,sigma: -2*Y/(2*sigma + 2)**2 )(*args)'
      },
      {
        'name': 'sigma',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Y/(sigma + 1)**3',
        'derivative_lambda': 'lambda args : (lambda Y,sigma: Y/(sigma + 1)**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Y',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'sigma',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman86',
    'DescriptiveName': 'Feynman86, Lecture III.4.32',
    'Constraints': [
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-omega*exp(h*omega/(2*pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -omega*np.exp(h*omega/(2*np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'omega**2*(-exp(h*omega/(2*pi*T*kb)) + 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(4*pi**2*T**2*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: omega**2*(-np.exp(h*omega/(2*np.pi*T*kb)) + 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(4*np.pi**2*T**2*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-h*exp(h*omega/(2*pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -h*np.exp(h*omega/(2*np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*(-exp(h*omega/(2*pi*T*kb)) + 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(4*pi**2*T**2*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h**2*(-np.exp(h*omega/(2*np.pi*T*kb)) + 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(4*np.pi**2*T**2*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h*omega*exp(h*omega/(2*pi*T*kb))/(2*pi*T*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(2*np.pi*T*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h*omega*(-exp(h*omega/(2*pi*T*kb)) - h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) + h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(pi*T*kb**3*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h*omega*(-np.exp(h*omega/(2*np.pi*T*kb)) - h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) + h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(np.pi*T*kb**3*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h*omega*exp(h*omega/(2*pi*T*kb))/(2*pi*T**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(2*np.pi*T**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h*omega*(-exp(h*omega/(2*pi*T*kb)) - h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) + h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(pi*T**3*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h*omega*(-np.exp(h*omega/(2*np.pi*T*kb)) - h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) + h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(np.pi*T**3*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman87',
    'DescriptiveName': 'Feynman87, Lecture III.4.33',
    'Constraints': [
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'omega/(2*pi*(exp(h*omega/(2*pi*T*kb)) - 1)) - h*omega**2*exp(h*omega/(2*pi*T*kb))/(4*pi**2*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: omega/(2*np.pi*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) - h*omega**2*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**2*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-omega**2*(4*exp(h*omega/(2*pi*T*kb)) + h*omega*(exp(h*omega/(2*pi*T*kb)) - 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(pi*T*kb))/(8*pi**2*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -omega**2*(4*np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*(np.exp(h*omega/(2*np.pi*T*kb)) - 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(np.pi*T*kb))/(8*np.pi**2*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'h/(2*pi*(exp(h*omega/(2*pi*T*kb)) - 1)) - h**2*omega*exp(h*omega/(2*pi*T*kb))/(4*pi**2*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h/(2*np.pi*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)) - h**2*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**2*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-h**2*(4*exp(h*omega/(2*pi*T*kb)) + h*omega*(exp(h*omega/(2*pi*T*kb)) - 2*exp(h*omega/(pi*T*kb))/(exp(h*omega/(2*pi*T*kb)) - 1))/(pi*T*kb))/(8*pi**2*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -h**2*(4*np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*(np.exp(h*omega/(2*np.pi*T*kb)) - 2*np.exp(h*omega/(np.pi*T*kb))/(np.exp(h*omega/(2*np.pi*T*kb)) - 1))/(np.pi*T*kb))/(8*np.pi**2*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*omega**2*exp(h*omega/(2*pi*T*kb))/(4*pi**2*T*kb**2*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h**2*omega**2*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**2*T*kb**2*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-h**2*omega**2*(exp(h*omega/(2*pi*T*kb)) + h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) - h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(2*pi**2*T*kb**3*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -h**2*omega**2*(np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) - h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(2*np.pi**2*T*kb**3*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*omega**2*exp(h*omega/(2*pi*T*kb))/(4*pi**2*T**2*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: h**2*omega**2*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi**2*T**2*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-h**2*omega**2*(exp(h*omega/(2*pi*T*kb)) + h*omega*exp(h*omega/(2*pi*T*kb))/(4*pi*T*kb) - h*omega*exp(h*omega/(pi*T*kb))/(2*pi*T*kb*(exp(h*omega/(2*pi*T*kb)) - 1)))/(2*pi**2*T**3*kb*(exp(h*omega/(2*pi*T*kb)) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda h,omega,kb,T: -h**2*omega**2*(np.exp(h*omega/(2*np.pi*T*kb)) + h*omega*np.exp(h*omega/(2*np.pi*T*kb))/(4*np.pi*T*kb) - h*omega*np.exp(h*omega/(np.pi*T*kb))/(2*np.pi*T*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)))/(2*np.pi**2*T**3*kb*(np.exp(h*omega/(2*np.pi*T*kb)) - 1)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman88',
    'DescriptiveName': 'Feynman88, Lecture III.7.38',
    'Constraints': [
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*B/h',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: 4*np.pi*B/h )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: 0 )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*mom/h',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: 4*np.pi*mom/h )(*args)'
      },
      {
        'name': 'B',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-4*pi*B*mom/h**2',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: -4*np.pi*B*mom/h**2 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*B*mom/h**3',
        'derivative_lambda': 'lambda args : (lambda mom,B,h: 8*np.pi*B*mom/h**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'B',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman89',
    'DescriptiveName': 'Feynman89, Lecture III.8.54',
    'Constraints': [
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '4*pi*t*sin(2*pi*E_n*t/h)*cos(2*pi*E_n*t/h)/h',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: 4*np.pi*t*np.sin(2*np.pi*E_n*t/h)*np.cos(2*np.pi*E_n*t/h)/h )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '8*pi**2*t**2*(-sin(2*pi*E_n*t/h)**2 + cos(2*pi*E_n*t/h)**2)/h**2',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: 8*np.pi**2*t**2*(-np.sin(2*np.pi*E_n*t/h)**2 + np.cos(2*np.pi*E_n*t/h)**2)/h**2 )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '4*pi*E_n*sin(2*pi*E_n*t/h)*cos(2*pi*E_n*t/h)/h',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: 4*np.pi*E_n*np.sin(2*np.pi*E_n*t/h)*np.cos(2*np.pi*E_n*t/h)/h )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '8*pi**2*E_n**2*(-sin(2*pi*E_n*t/h)**2 + cos(2*pi*E_n*t/h)**2)/h**2',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: 8*np.pi**2*E_n**2*(-np.sin(2*np.pi*E_n*t/h)**2 + np.cos(2*np.pi*E_n*t/h)**2)/h**2 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-4*pi*E_n*t*sin(2*pi*E_n*t/h)*cos(2*pi*E_n*t/h)/h**2',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: -4*np.pi*E_n*t*np.sin(2*np.pi*E_n*t/h)*np.cos(2*np.pi*E_n*t/h)/h**2 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '8*pi*E_n*t*(-pi*E_n*t*sin(2*pi*E_n*t/h)**2/h + pi*E_n*t*cos(2*pi*E_n*t/h)**2/h + sin(2*pi*E_n*t/h)*cos(2*pi*E_n*t/h))/h**3',
        'derivative_lambda': 'lambda args : (lambda E_n,t,h: 8*np.pi*E_n*t*(-np.pi*E_n*t*np.sin(2*np.pi*E_n*t/h)**2/h + np.pi*E_n*t*np.cos(2*np.pi*E_n*t/h)**2/h + np.sin(2*np.pi*E_n*t/h)*np.cos(2*np.pi*E_n*t/h))/h**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 't',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 4.0
      }
    ]
  },
  {
    'EquationName': 'Feynman90',
    'DescriptiveName': 'Feynman90, Lecture III.9.52',
    'Constraints': [
      {
        'name': 'p_d',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*Ef*sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 8*np.pi*Ef*np.sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 'p_d',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 0 )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*p_d*sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 8*np.pi*p_d*np.sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 0 )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '16*pi*Ef*p_d*(omega/2 - omega_0/2)*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(h*t*(omega - omega_0)**2) - 8*pi*Ef*p_d*sin(t*(omega - omega_0)/2)**2/(h*t**2*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 16*np.pi*Ef*p_d*(omega/2 - omega_0/2)*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(h*t*(omega - omega_0)**2) - 8*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)**2/(h*t**2*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 't',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '4*pi*Ef*p_d*(-sin(t*(omega - omega_0)/2)**2 + cos(t*(omega - omega_0)/2)**2 - 4*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(t*(omega - omega_0)) + 4*sin(t*(omega - omega_0)/2)**2/(t**2*(omega - omega_0)**2))/(h*t)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 4*np.pi*Ef*p_d*(-np.sin(t*(omega - omega_0)/2)**2 + np.cos(t*(omega - omega_0)/2)**2 - 4*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(t*(omega - omega_0)) + 4*np.sin(t*(omega - omega_0)/2)**2/(t**2*(omega - omega_0)**2))/(h*t) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-8*pi*Ef*p_d*sin(t*(omega - omega_0)/2)**2/(h**2*t*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: -8*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)**2/(h**2*t*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '16*pi*Ef*p_d*sin(t*(omega - omega_0)/2)**2/(h**3*t*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 16*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)**2/(h**3*t*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '8*pi*Ef*p_d*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(h*(omega - omega_0)**2) - 16*pi*Ef*p_d*sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**3)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 8*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(h*(omega - omega_0)**2) - 16*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**3) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '4*pi*Ef*p_d*(-t*(sin(t*(omega - omega_0)/2)**2 - cos(t*(omega - omega_0)/2)**2) - 8*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(omega - omega_0) + 12*sin(t*(omega - omega_0)/2)**2/(t*(omega - omega_0)**2))/(h*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 4*np.pi*Ef*p_d*(-t*(np.sin(t*(omega - omega_0)/2)**2 - np.cos(t*(omega - omega_0)/2)**2) - 8*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(omega - omega_0) + 12*np.sin(t*(omega - omega_0)/2)**2/(t*(omega - omega_0)**2))/(h*(omega - omega_0)**2) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-8*pi*Ef*p_d*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(h*(omega - omega_0)**2) + 16*pi*Ef*p_d*sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**3)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: -8*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(h*(omega - omega_0)**2) + 16*np.pi*Ef*p_d*np.sin(t*(omega - omega_0)/2)**2/(h*t*(omega - omega_0)**3) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '4*pi*Ef*p_d*(-t*(sin(t*(omega - omega_0)/2)**2 - cos(t*(omega - omega_0)/2)**2) - 8*sin(t*(omega - omega_0)/2)*cos(t*(omega - omega_0)/2)/(omega - omega_0) + 12*sin(t*(omega - omega_0)/2)**2/(t*(omega - omega_0)**2))/(h*(omega - omega_0)**2)',
        'derivative_lambda': 'lambda args : (lambda p_d,Ef,t,h,omega,omega_0: 4*np.pi*Ef*p_d*(-t*(np.sin(t*(omega - omega_0)/2)**2 - np.cos(t*(omega - omega_0)/2)**2) - 8*np.sin(t*(omega - omega_0)/2)*np.cos(t*(omega - omega_0)/2)/(omega - omega_0) + 12*np.sin(t*(omega - omega_0)/2)**2/(t*(omega - omega_0)**2))/(h*(omega - omega_0)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'p_d',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 't',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'omega_0',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman91',
    'DescriptiveName': 'Feynman91, Lecture III.10.19',
    'Constraints': [
      {
        'name': 'mom',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'mom',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: 0 )(*args)'
      },
      {
        'name': 'Bx',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Bx*mom/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: Bx*mom/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'Bx',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-mom*(Bx**2/(Bx**2 + By**2 + Bz**2) - 1)/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: -mom*(Bx**2/(Bx**2 + By**2 + Bz**2) - 1)/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'By',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'By*mom/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: By*mom/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'By',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-mom*(By**2/(Bx**2 + By**2 + Bz**2) - 1)/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: -mom*(By**2/(Bx**2 + By**2 + Bz**2) - 1)/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'Bz',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Bz*mom/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: Bz*mom/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      },
      {
        'name': 'Bz',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-mom*(Bz**2/(Bx**2 + By**2 + Bz**2) - 1)/sqrt(Bx**2 + By**2 + Bz**2)',
        'derivative_lambda': 'lambda args : (lambda mom,Bx,By,Bz: -mom*(Bz**2/(Bx**2 + By**2 + Bz**2) - 1)/np.sqrt(Bx**2 + By**2 + Bz**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'mom',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Bx',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'By',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Bz',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman92',
    'DescriptiveName': 'Feynman92, Lecture III.12.43',
    'Constraints': [
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h/(2*pi)',
        'derivative_lambda': 'lambda args : (lambda n,h: h/(2*np.pi) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,h: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'n/(2*pi)',
        'derivative_lambda': 'lambda args : (lambda n,h: n/(2*np.pi) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda n,h: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman93',
    'DescriptiveName': 'Feynman93, Lecture III.13.18',
    'Constraints': [
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*d**2*k/h',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 4*np.pi*d**2*k/h )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 0 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*E_n*d*k/h',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 8*np.pi*E_n*d*k/h )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*E_n*k/h',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 8*np.pi*E_n*k/h )(*args)'
      },
      {
        'name': 'k',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*E_n*d**2/h',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 4*np.pi*E_n*d**2/h )(*args)'
      },
      {
        'name': 'k',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 0 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-4*pi*E_n*d**2*k/h**2',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: -4*np.pi*E_n*d**2*k/h**2 )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '8*pi*E_n*d**2*k/h**3',
        'derivative_lambda': 'lambda args : (lambda E_n,d,k,h: 8*np.pi*E_n*d**2*k/h**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'k',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman94',
    'DescriptiveName': 'Feynman94, Lecture III.14.14',
    'Constraints': [
      {
        'name': 'I_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'exp(Volt*q/(T*kb)) - 1',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: np.exp(Volt*q/(T*kb)) - 1 )(*args)'
      },
      {
        'name': 'I_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: 0 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*Volt*exp(Volt*q/(T*kb))/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*Volt*np.exp(Volt*q/(T*kb))/(T*kb) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*Volt**2*exp(Volt*q/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*Volt**2*np.exp(Volt*q/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*q*exp(Volt*q/(T*kb))/(T*kb)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*q*np.exp(Volt*q/(T*kb))/(T*kb) )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*q**2*exp(Volt*q/(T*kb))/(T**2*kb**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*q**2*np.exp(Volt*q/(T*kb))/(T**2*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-I_0*Volt*q*exp(Volt*q/(T*kb))/(T*kb**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: -I_0*Volt*q*np.exp(Volt*q/(T*kb))/(T*kb**2) )(*args)'
      },
      {
        'name': 'kb',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*Volt*q*(2 + Volt*q/(T*kb))*exp(Volt*q/(T*kb))/(T*kb**3)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*Volt*q*(2 + Volt*q/(T*kb))*np.exp(Volt*q/(T*kb))/(T*kb**3) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-I_0*Volt*q*exp(Volt*q/(T*kb))/(T**2*kb)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: -I_0*Volt*q*np.exp(Volt*q/(T*kb))/(T**2*kb) )(*args)'
      },
      {
        'name': 'T',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'I_0*Volt*q*(2 + Volt*q/(T*kb))*exp(Volt*q/(T*kb))/(T**3*kb)',
        'derivative_lambda': 'lambda args : (lambda I_0,q,Volt,kb,T: I_0*Volt*q*(2 + Volt*q/(T*kb))*np.exp(Volt*q/(T*kb))/(T**3*kb) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'I_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'Volt',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'kb',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'T',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Feynman95',
    'DescriptiveName': 'Feynman95, Lecture III.15.12',
    'Constraints': [
      {
        'name': 'U',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2 - 2*cos(d*k)',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 2 - 2*np.cos(d*k) )(*args)'
      },
      {
        'name': 'U',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 0 )(*args)'
      },
      {
        'name': 'k',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '2*U*d*sin(d*k)',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 2*U*d*np.sin(d*k) )(*args)'
      },
      {
        'name': 'k',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*U*d**2*cos(d*k)',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 2*U*d**2*np.cos(d*k) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '2*U*k*sin(d*k)',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 2*U*k*np.sin(d*k) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*U*k**2*cos(d*k)',
        'derivative_lambda': 'lambda args : (lambda U,k,d: 2*U*k**2*np.cos(d*k) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'U',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'k',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman96',
    'DescriptiveName': 'Feynman96, Lecture III.15.14',
    'Constraints': [
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h/(4*pi**2*E_n*d**2)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: h/(4*np.pi**2*E_n*d**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi**2*E_n*d**2)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: 1/(4*np.pi**2*E_n*d**2) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-h**2/(8*pi**2*E_n**2*d**2)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: -h**2/(8*np.pi**2*E_n**2*d**2) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2/(4*pi**2*E_n**3*d**2)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: h**2/(4*np.pi**2*E_n**3*d**2) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-h**2/(4*pi**2*E_n*d**3)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: -h**2/(4*np.pi**2*E_n*d**3) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*h**2/(4*pi**2*E_n*d**4)',
        'derivative_lambda': 'lambda args : (lambda h,E_n,d: 3*h**2/(4*np.pi**2*E_n*d**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman97',
    'DescriptiveName': 'Feynman97, Lecture III.15.27',
    'Constraints': [
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*pi/(d*n)',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: 2*np.pi/(d*n) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: 0 )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*pi*alpha/(d*n**2)',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: -2*np.pi*alpha/(d*n**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*alpha/(d*n**3)',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: 4*np.pi*alpha/(d*n**3) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*pi*alpha/(d**2*n)',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: -2*np.pi*alpha/(d**2*n) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*alpha/(d**3*n)',
        'derivative_lambda': 'lambda args : (lambda alpha,n,d: 4*np.pi*alpha/(d**3*n) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman98',
    'DescriptiveName': 'Feynman98, Lecture III.17.37',
    'Constraints': [
      {
        'name': 'beta',
        'order_derivative': 1,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: 0 )(*args)'
      },
      {
        'name': 'beta',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: 0 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'beta*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: beta*np.cos(theta) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-alpha*beta*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: -alpha*beta*np.sin(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-alpha*beta*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda beta,alpha,theta: -alpha*beta*np.cos(theta) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'beta',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman99',
    'DescriptiveName': 'Feynman99, Lecture III.19.51',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q**4/(8*epsilon**2*h**2*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -q**4/(8*epsilon**2*h**2*n**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: 0 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-m*q**3/(2*epsilon**2*h**2*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -m*q**3/(2*epsilon**2*h**2*n**2) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*m*q**2/(2*epsilon**2*h**2*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -3*m*q**2/(2*epsilon**2*h**2*n**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*q**4/(4*epsilon**2*h**3*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: m*q**4/(4*epsilon**2*h**3*n**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*m*q**4/(4*epsilon**2*h**4*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -3*m*q**4/(4*epsilon**2*h**4*n**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*q**4/(4*epsilon**2*h**2*n**3)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: m*q**4/(4*epsilon**2*h**2*n**3) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*m*q**4/(4*epsilon**2*h**2*n**4)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -3*m*q**4/(4*epsilon**2*h**2*n**4) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*q**4/(4*epsilon**3*h**2*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: m*q**4/(4*epsilon**3*h**2*n**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*m*q**4/(4*epsilon**4*h**2*n**2)',
        'derivative_lambda': 'lambda args : (lambda m,q,h,n,epsilon: -3*m*q**4/(4*epsilon**4*h**2*n**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Feynman100',
    'DescriptiveName': 'Feynman100, Lecture III.21.20',
    'Constraints': [
      {
        'name': 'rho_c_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-A_vec*q/m',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: -A_vec*q/m )(*args)'
      },
      {
        'name': 'rho_c_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: 0 )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-A_vec*rho_c_0/m',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: -A_vec*rho_c_0/m )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: 0 )(*args)'
      },
      {
        'name': 'A_vec',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q*rho_c_0/m',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: -q*rho_c_0/m )(*args)'
      },
      {
        'name': 'A_vec',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: 0 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'A_vec*q*rho_c_0/m**2',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: A_vec*q*rho_c_0/m**2 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*A_vec*q*rho_c_0/m**3',
        'derivative_lambda': 'lambda args : (lambda rho_c_0,q,A_vec,m: -2*A_vec*q*rho_c_0/m**3 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'rho_c_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'A_vec',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus1',
    'DescriptiveName': 'Bonus1.0, Rutherford scattering',
    'Constraints': [
      {
        'name': 'Z_1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'Z_1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'Z_2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2*alpha**2*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2*alpha**2*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'Z_2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*alpha**2*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*alpha**2*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*c**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*c**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'hbar',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha**2*c**2*hbar/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha**2*c**2*hbar/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'hbar',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha**2*c**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha**2*c**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha**2*c*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha**2*c*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha**2*hbar**2/(8*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha**2*hbar**2/(8*E_n**2*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Z_1**2*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**3*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: -Z_1**2*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**3*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*Z_1**2*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**4*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: 3*Z_1**2*Z_2**2*alpha**2*c**2*hbar**2/(8*E_n**4*np.sin(theta/2)**4) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-Z_1**2*Z_2**2*alpha**2*c**2*hbar**2*cos(theta/2)/(8*E_n**2*sin(theta/2)**5)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: -Z_1**2*Z_2**2*alpha**2*c**2*hbar**2*np.cos(theta/2)/(8*E_n**2*np.sin(theta/2)**5) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Z_1**2*Z_2**2*alpha**2*c**2*hbar**2*(1 + 5*cos(theta/2)**2/sin(theta/2)**2)/(16*E_n**2*sin(theta/2)**4)',
        'derivative_lambda': 'lambda args : (lambda Z_1,Z_2,alpha,hbar,c,E_n,theta: Z_1**2*Z_2**2*alpha**2*c**2*hbar**2*(1 + 5*np.cos(theta/2)**2/np.sin(theta/2)**2)/(16*E_n**2*np.sin(theta/2)**4) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Z_1',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'Z_2',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'hbar',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus2',
    'DescriptiveName': 'Bonus2.0, 3.55 Goldstein',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-E_n*cos(theta1 - theta2)/(k_G*m*sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + k_G*(sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2) + 1)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -E_n*np.cos(theta1 - theta2)/(k_G*m*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + k_G*(np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2) + 1)/L**2 )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-E_n**2*L**2*cos(theta1 - theta2)/(k_G**3*m**3*(2*E_n*L**2/(k_G**2*m) + 1)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -E_n**2*L**2*np.cos(theta1 - theta2)/(k_G**3*m**3*(2*E_n*L**2/(k_G**2*m) + 1)**(3/2)) )(*args)'
      },
      {
        'name': 'k_G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-2*E_n*cos(theta1 - theta2)/(k_G**2*sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + m*(sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2) + 1)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -2*E_n*np.cos(theta1 - theta2)/(k_G**2*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + m*(np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2) + 1)/L**2 )(*args)'
      },
      {
        'name': 'k_G',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*E_n*(2*E_n*L**2/(k_G**2*m*(2*E_n*L**2/(k_G**2*m) + 1)) - 1)*cos(theta1 - theta2)/(k_G**3*sqrt(2*E_n*L**2/(k_G**2*m) + 1))',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -2*E_n*(2*E_n*L**2/(k_G**2*m*(2*E_n*L**2/(k_G**2*m) + 1)) - 1)*np.cos(theta1 - theta2)/(k_G**3*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '2*E_n*cos(theta1 - theta2)/(L*k_G*sqrt(2*E_n*L**2/(k_G**2*m) + 1)) - 2*k_G*m*(sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2) + 1)/L**3',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: 2*E_n*np.cos(theta1 - theta2)/(L*k_G*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) - 2*k_G*m*(np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2) + 1)/L**3 )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*(-E_n*(2*E_n*L**2/(k_G**2*m*(2*E_n*L**2/(k_G**2*m) + 1)) - 1)*cos(theta1 - theta2)/(k_G*sqrt(2*E_n*L**2/(k_G**2*m) + 1)) - 4*E_n*cos(theta1 - theta2)/(k_G*sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + 3*k_G*m*(sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2) + 1)/L**2)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: 2*(-E_n*(2*E_n*L**2/(k_G**2*m*(2*E_n*L**2/(k_G**2*m) + 1)) - 1)*np.cos(theta1 - theta2)/(k_G*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) - 4*E_n*np.cos(theta1 - theta2)/(k_G*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) + 3*k_G*m*(np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2) + 1)/L**2)/L**2 )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'cos(theta1 - theta2)/(k_G*sqrt(2*E_n*L**2/(k_G**2*m) + 1))',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: np.cos(theta1 - theta2)/(k_G*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-L**2*cos(theta1 - theta2)/(k_G**3*m*(2*E_n*L**2/(k_G**2*m) + 1)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -L**2*np.cos(theta1 - theta2)/(k_G**3*m*(2*E_n*L**2/(k_G**2*m) + 1)**(3/2)) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-k_G*m*sqrt(2*E_n*L**2/(k_G**2*m) + 1)*sin(theta1 - theta2)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -k_G*m*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.sin(theta1 - theta2)/L**2 )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-k_G*m*sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -k_G*m*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2)/L**2 )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'k_G*m*sqrt(2*E_n*L**2/(k_G**2*m) + 1)*sin(theta1 - theta2)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: k_G*m*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.sin(theta1 - theta2)/L**2 )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-k_G*m*sqrt(2*E_n*L**2/(k_G**2*m) + 1)*cos(theta1 - theta2)/L**2',
        'derivative_lambda': 'lambda args : (lambda m,k_G,L,E_n,theta1,theta2: -k_G*m*np.sqrt(2*E_n*L**2/(k_G**2*m) + 1)*np.cos(theta1 - theta2)/L**2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'k_G',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'L',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta1',
        'low': 0.0,
        'high': 6.0
      },
      {
        'name': 'theta2',
        'low': 0.0,
        'high': 6.0
      }
    ]
  },
  {
    'EquationName': 'Bonus3',
    'DescriptiveName': 'Bonus3.0, 3.64 Goldstein',
    'Constraints': [
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '(1 - alpha**2)/(alpha*cos(theta1 - theta2) + 1)',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: (1 - alpha**2)/(alpha*np.cos(theta1 - theta2) + 1) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: 0 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*alpha*d/(alpha*cos(theta1 - theta2) + 1) - d*(1 - alpha**2)*cos(theta1 - theta2)/(alpha*cos(theta1 - theta2) + 1)**2',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: -2*alpha*d/(alpha*np.cos(theta1 - theta2) + 1) - d*(1 - alpha**2)*np.cos(theta1 - theta2)/(alpha*np.cos(theta1 - theta2) + 1)**2 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '2*d*(2*alpha*cos(theta1 - theta2)/(alpha*cos(theta1 - theta2) + 1) - (alpha**2 - 1)*cos(theta1 - theta2)**2/(alpha*cos(theta1 - theta2) + 1)**2 - 1)/(alpha*cos(theta1 - theta2) + 1)',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: 2*d*(2*alpha*np.cos(theta1 - theta2)/(alpha*np.cos(theta1 - theta2) + 1) - (alpha**2 - 1)*np.cos(theta1 - theta2)**2/(alpha*np.cos(theta1 - theta2) + 1)**2 - 1)/(alpha*np.cos(theta1 - theta2) + 1) )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'alpha*d*(1 - alpha**2)*sin(theta1 - theta2)/(alpha*cos(theta1 - theta2) + 1)**2',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: alpha*d*(1 - alpha**2)*np.sin(theta1 - theta2)/(alpha*np.cos(theta1 - theta2) + 1)**2 )(*args)'
      },
      {
        'name': 'theta1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*d*(alpha**2 - 1)*(2*alpha*sin(theta1 - theta2)**2/(alpha*cos(theta1 - theta2) + 1) + cos(theta1 - theta2))/(alpha*cos(theta1 - theta2) + 1)**2',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: -alpha*d*(alpha**2 - 1)*(2*alpha*np.sin(theta1 - theta2)**2/(alpha*np.cos(theta1 - theta2) + 1) + np.cos(theta1 - theta2))/(alpha*np.cos(theta1 - theta2) + 1)**2 )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-alpha*d*(1 - alpha**2)*sin(theta1 - theta2)/(alpha*cos(theta1 - theta2) + 1)**2',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: -alpha*d*(1 - alpha**2)*np.sin(theta1 - theta2)/(alpha*np.cos(theta1 - theta2) + 1)**2 )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*d*(alpha**2 - 1)*(2*alpha*sin(theta1 - theta2)**2/(alpha*cos(theta1 - theta2) + 1) + cos(theta1 - theta2))/(alpha*cos(theta1 - theta2) + 1)**2',
        'derivative_lambda': 'lambda args : (lambda d,alpha,theta1,theta2: -alpha*d*(alpha**2 - 1)*(2*alpha*np.sin(theta1 - theta2)**2/(alpha*np.cos(theta1 - theta2) + 1) + np.cos(theta1 - theta2))/(alpha*np.cos(theta1 - theta2) + 1)**2 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'd',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'alpha',
        'low': 2.0,
        'high': 4.0
      },
      {
        'name': 'theta1',
        'low': 4.0,
        'high': 5.0
      },
      {
        'name': 'theta2',
        'low': 4.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus4',
    'DescriptiveName': 'Bonus4.0, 3.16 Goldstein',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'sqrt(2)*m*sqrt((E_n - L**2/(2*m*r**2) - U)/m)*(L**2/(4*m**3*r**2) - (E_n - L**2/(2*m*r**2) - U)/(2*m**2))/(E_n - L**2/(2*m*r**2) - U)',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: np.sqrt(2)*m*np.sqrt((E_n - L**2/(2*m*r**2) - U)/m)*(L**2/(4*m**3*r**2) - (E_n - L**2/(2*m*r**2) - U)/(2*m**2))/(E_n - L**2/(2*m*r**2) - U) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'sqrt(2)*sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(-E_n + 2*L**2/(m*r**2) - L**2*(-E_n + L**2/(m*r**2) + U)/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) + U + (-E_n + L**2/(m*r**2) + U)**2/(-2*E_n + L**2/(m*r**2) + 2*U))/(m**2*(-2*E_n + L**2/(m*r**2) + 2*U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: np.sqrt(2)*np.sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(-E_n + 2*L**2/(m*r**2) - L**2*(-E_n + L**2/(m*r**2) + U)/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) + U + (-E_n + L**2/(m*r**2) + U)**2/(-2*E_n + L**2/(m*r**2) + 2*U))/(m**2*(-2*E_n + L**2/(m*r**2) + 2*U)) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(2)*sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*(E_n - L**2/(2*m*r**2) - U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: np.sqrt(2)*np.sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*(E_n - L**2/(2*m*r**2) - U)) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)/(4*(-E_n + L**2/(2*m*r**2) + U)**2)',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*np.sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)/(4*(-E_n + L**2/(2*m*r**2) + U)**2) )(*args)'
      },
      {
        'name': 'U',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*(E_n - L**2/(2*m*r**2) - U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*np.sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*(E_n - L**2/(2*m*r**2) - U)) )(*args)'
      },
      {
        'name': 'U',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)/(4*(-E_n + L**2/(2*m*r**2) + U)**2)',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*np.sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)/(4*(-E_n + L**2/(2*m*r**2) + U)**2) )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*L*sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*m*r**2*(E_n - L**2/(2*m*r**2) - U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*L*np.sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*m*r**2*(E_n - L**2/(2*m*r**2) - U)) )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(L**2/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) - 1)/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*np.sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(L**2/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) - 1)/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(2)*L**2*sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*m*r**3*(E_n - L**2/(2*m*r**2) - U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: np.sqrt(2)*L**2*np.sqrt((E_n - L**2/(2*m*r**2) - U)/m)/(2*m*r**3*(E_n - L**2/(2*m*r**2) - U)) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-sqrt(2)*L**2*sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(L**2/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) - 3)/(m*r**4*(-2*E_n + L**2/(m*r**2) + 2*U))',
        'derivative_lambda': 'lambda args : (lambda m,E_n,U,L,r: -np.sqrt(2)*L**2*np.sqrt(-(-E_n + L**2/(2*m*r**2) + U)/m)*(L**2/(m*r**2*(-2*E_n + L**2/(m*r**2) + 2*U)) - 3)/(m*r**4*(-2*E_n + L**2/(m*r**2) + 2*U)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'E_n',
        'low': 8.0,
        'high': 12.0
      },
      {
        'name': 'U',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'L',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus5',
    'DescriptiveName': 'Bonus5.0, 3.74 Goldstein',
    'Constraints': [
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pi*sqrt(d)/sqrt(G*(m1 + m2))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 3*np.pi*np.sqrt(d)/np.sqrt(G*(m1 + m2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pi/(2*sqrt(d)*sqrt(G*(m1 + m2)))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 3*np.pi/(2*np.sqrt(d)*np.sqrt(G*(m1 + m2))) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '2*pi*d**(3/2)*(-m1/2 - m2/2)/(G*sqrt(G*(m1 + m2))*(m1 + m2))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 2*np.pi*d**(3/2)*(-m1/2 - m2/2)/(G*np.sqrt(G*(m1 + m2))*(m1 + m2)) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pi*d**(3/2)/(2*G**2*sqrt(G*(m1 + m2)))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 3*np.pi*d**(3/2)/(2*G**2*np.sqrt(G*(m1 + m2))) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-pi*d**(3/2)/(sqrt(G*(m1 + m2))*(m1 + m2))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: -np.pi*d**(3/2)/(np.sqrt(G*(m1 + m2))*(m1 + m2)) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pi*d**(3/2)/(2*sqrt(G*(m1 + m2))*(m1 + m2)**2)',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 3*np.pi*d**(3/2)/(2*np.sqrt(G*(m1 + m2))*(m1 + m2)**2) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-pi*d**(3/2)/(sqrt(G*(m1 + m2))*(m1 + m2))',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: -np.pi*d**(3/2)/(np.sqrt(G*(m1 + m2))*(m1 + m2)) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*pi*d**(3/2)/(2*sqrt(G*(m1 + m2))*(m1 + m2)**2)',
        'derivative_lambda': 'lambda args : (lambda d,G,m1,m2: 3*np.pi*d**(3/2)/(2*np.sqrt(G*(m1 + m2))*(m1 + m2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'd',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'G',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'm1',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'm2',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus6',
    'DescriptiveName': 'Bonus6.0, 3.99 Goldstein',
    'Constraints': [
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*L**2*epsilon/(Z_1**2*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*L**2*epsilon/(Z_1**2*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*L**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 1)/(Z_1**2*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*L**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 1)/(Z_1**2*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*L*epsilon**2/(Z_1**2*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*L*epsilon**2/(Z_1**2*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'L',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 1)/(Z_1**2*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 1)/(Z_1**2*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m**2*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: -E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m**2*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'E_n*L**2*epsilon**2*(-E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 2)/(Z_1**2*Z_2**2*m**3*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: E_n*L**2*epsilon**2*(-E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 2)/(Z_1**2*Z_2**2*m**3*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'Z_1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*E_n*L**2*epsilon**2/(Z_1**3*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: -2*E_n*L**2*epsilon**2/(Z_1**3*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'Z_1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*L**2*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 3)/(Z_1**4*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*L**2*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 3)/(Z_1**4*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'Z_2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**3*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: -2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**3*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'Z_2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n*L**2*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 3)/(Z_1**2*Z_2**4*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 2*E_n*L**2*epsilon**2*(-2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 3)/(Z_1**2*Z_2**4*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-4*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**5*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: -4*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**5*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*E_n*L**2*epsilon**2*(-4*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 5)/(Z_1**2*Z_2**2*m*q**6*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: 4*E_n*L**2*epsilon**2*(-4*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) + 5)/(Z_1**2*Z_2**2*m*q**6*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4*np.sqrt(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-L**4*epsilon**4/(Z_1**4*Z_2**4*m**2*q**8*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda epsilon,L,m,Z_1,Z_2,q,E_n: -L**4*epsilon**4/(Z_1**4*Z_2**4*m**2*q**8*(2*E_n*L**2*epsilon**2/(Z_1**2*Z_2**2*m*q**4) + 1)**(3/2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'L',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Z_1',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Z_2',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus7',
    'DescriptiveName': 'Bonus7.0, Friedman Equation',
    'Constraints': [
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*rho/(3*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: 4*np.pi*rho/(3*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-16*pi**2*rho**2/(9*(8*pi*G*rho/3 - alpha*c**2/d**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -16*np.pi**2*rho**2/(9*(8*np.pi*G*rho/3 - alpha*c**2/d**2)**(3/2)) )(*args)'
      },
      {
        'name': 'rho',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*pi*G/(3*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: 4*np.pi*G/(3*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'rho',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-16*pi**2*G**2/(9*(8*pi*G*rho/3 - alpha*c**2/d**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -16*np.pi**2*G**2/(9*(8*np.pi*G*rho/3 - alpha*c**2/d**2)**(3/2)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-c**2/(2*d**2*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -c**2/(2*d**2*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-c**4/(4*d**4*(8*pi*G*rho/3 - alpha*c**2/d**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -c**4/(4*d**4*(8*np.pi*G*rho/3 - alpha*c**2/d**2)**(3/2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*c/(d**2*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -alpha*c/(d**2*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*(alpha*c**2/(d**2*(8*pi*G*rho/3 - alpha*c**2/d**2)) + 1)/(d**2*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -alpha*(alpha*c**2/(d**2*(8*np.pi*G*rho/3 - alpha*c**2/d**2)) + 1)/(d**2*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*c**2/(d**3*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: alpha*c**2/(d**3*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*c**2*(alpha*c**2/(d**2*(8*pi*G*rho/3 - alpha*c**2/d**2)) + 3)/(d**4*sqrt(8*pi*G*rho/3 - alpha*c**2/d**2))',
        'derivative_lambda': 'lambda args : (lambda G,rho,alpha,c,d: -alpha*c**2*(alpha*c**2/(d**2*(8*np.pi*G*rho/3 - alpha*c**2/d**2)) + 3)/(d**4*np.sqrt(8*np.pi*G*rho/3 - alpha*c**2/d**2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'G',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'rho',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus8',
    'DescriptiveName': 'Bonus8.0, Compton Scattering',
    'Constraints': [
      {
        'name': 'E_n',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-E_n*(1 - cos(theta))/(c**2*m*(E_n*(1 - cos(theta))/(c**2*m) + 1)**2) + 1/(E_n*(1 - cos(theta))/(c**2*m) + 1)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: -E_n*(1 - np.cos(theta))/(c**2*m*(E_n*(1 - np.cos(theta))/(c**2*m) + 1)**2) + 1/(E_n*(1 - np.cos(theta))/(c**2*m) + 1) )(*args)'
      },
      {
        'name': 'E_n',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '2*(-E_n*(cos(theta) - 1)/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)) + 1)*(cos(theta) - 1)/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: 2*(-E_n*(np.cos(theta) - 1)/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)) + 1)*(np.cos(theta) - 1)/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'E_n**2*(1 - cos(theta))/(c**2*m**2*(E_n*(1 - cos(theta))/(c**2*m) + 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: E_n**2*(1 - np.cos(theta))/(c**2*m**2*(E_n*(1 - np.cos(theta))/(c**2*m) + 1)**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-2*E_n**2*(E_n*(cos(theta) - 1)/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)) - 1)*(cos(theta) - 1)/(c**2*m**3*(E_n*(cos(theta) - 1)/(c**2*m) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: -2*E_n**2*(E_n*(np.cos(theta) - 1)/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)) - 1)*(np.cos(theta) - 1)/(c**2*m**3*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '2*E_n**2*(1 - cos(theta))/(c**3*m*(E_n*(1 - cos(theta))/(c**2*m) + 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: 2*E_n**2*(1 - np.cos(theta))/(c**3*m*(E_n*(1 - np.cos(theta))/(c**2*m) + 1)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*E_n**2*(4*E_n*(cos(theta) - 1)/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)) - 3)*(cos(theta) - 1)/(c**4*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: -2*E_n**2*(4*E_n*(np.cos(theta) - 1)/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)) - 3)*(np.cos(theta) - 1)/(c**4*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)**2) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-E_n**2*sin(theta)/(c**2*m*(E_n*(1 - cos(theta))/(c**2*m) + 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: -E_n**2*np.sin(theta)/(c**2*m*(E_n*(1 - np.cos(theta))/(c**2*m) + 1)**2) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-E_n**2*(2*E_n*sin(theta)**2/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)) + cos(theta))/(c**2*m*(E_n*(cos(theta) - 1)/(c**2*m) - 1)**2)',
        'derivative_lambda': 'lambda args : (lambda E_n,m,c,theta: -E_n**2*(2*E_n*np.sin(theta)**2/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)) + np.cos(theta))/(c**2*m*(E_n*(np.cos(theta) - 1)/(c**2*m) - 1)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'E_n',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus9',
    'DescriptiveName': 'Bonus9.0, Gravitational wave ratiated power',
    'Constraints': [
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-128*G**3*m1**2*m2**2*(m1 + m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -128*G**3*m1**2*m2**2*(m1 + m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-384*G**2*m1**2*m2**2*(m1 + m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -384*G**2*m1**2*m2**2*(m1 + m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '32*G**4*m1**2*m2**2*(m1 + m2)/(c**6*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: 32*G**4*m1**2*m2**2*(m1 + m2)/(c**6*r**5) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-192*G**4*m1**2*m2**2*(m1 + m2)/(c**7*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -192*G**4*m1**2*m2**2*(m1 + m2)/(c**7*r**5) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-32*G**4*m1**2*m2**2/(5*c**5*r**5) - 64*G**4*m1*m2**2*(m1 + m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -32*G**4*m1**2*m2**2/(5*c**5*r**5) - 64*G**4*m1*m2**2*(m1 + m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'm1',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-64*G**4*m2**2*(3*m1 + m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -64*G**4*m2**2*(3*m1 + m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-32*G**4*m1**2*m2**2/(5*c**5*r**5) - 64*G**4*m1**2*m2*(m1 + m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -32*G**4*m1**2*m2**2/(5*c**5*r**5) - 64*G**4*m1**2*m2*(m1 + m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'm2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-64*G**4*m1**2*(m1 + 3*m2)/(5*c**5*r**5)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -64*G**4*m1**2*(m1 + 3*m2)/(5*c**5*r**5) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '32*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**6)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: 32*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**6) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-192*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**7)',
        'derivative_lambda': 'lambda args : (lambda G,c,m1,m2,r: -192*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**7) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'G',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 2.0
      },
      {
        'name': 'm1',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm2',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Bonus10',
    'DescriptiveName': 'Bonus10.0, Relativistic aberation',
    'Constraints': [
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-(v/(c**2*(1 - v*cos(theta2)/c)) - v*(cos(theta2) - v/c)*cos(theta2)/(c**2*(1 - v*cos(theta2)/c)**2))/sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: -(v/(c**2*(1 - v*np.cos(theta2)/c)) - v*(np.cos(theta2) - v/c)*np.cos(theta2)/(c**2*(1 - v*np.cos(theta2)/c)**2))/np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'v*(2 - 2*(cos(theta2) - v/c)*cos(theta2)/(1 - v*cos(theta2)/c) + 2*v*cos(theta2)/(c*(1 - v*cos(theta2)/c)) - 2*v*(cos(theta2) - v/c)*cos(theta2)**2/(c*(1 - v*cos(theta2)/c)**2) - v*(1 - (cos(theta2) - v/c)*cos(theta2)/(1 - v*cos(theta2)/c))**2*(cos(theta2) - v/c)/(c*(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c)**2))/(c**3*sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: v*(2 - 2*(np.cos(theta2) - v/c)*np.cos(theta2)/(1 - v*np.cos(theta2)/c) + 2*v*np.cos(theta2)/(c*(1 - v*np.cos(theta2)/c)) - 2*v*(np.cos(theta2) - v/c)*np.cos(theta2)**2/(c*(1 - v*np.cos(theta2)/c)**2) - v*(1 - (np.cos(theta2) - v/c)*np.cos(theta2)/(1 - v*np.cos(theta2)/c))**2*(np.cos(theta2) - v/c)/(c*(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)**2))/(c**3*np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-(-1/(c*(1 - v*cos(theta2)/c)) + (cos(theta2) - v/c)*cos(theta2)/(c*(1 - v*cos(theta2)/c)**2))/sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: -(-1/(c*(1 - v*np.cos(theta2)/c)) + (np.cos(theta2) - v/c)*np.cos(theta2)/(c*(1 - v*np.cos(theta2)/c)**2))/np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '(1 - (cos(theta2) - v/c)*cos(theta2)/(1 - v*cos(theta2)/c))*(2*cos(theta2) - (1 - (cos(theta2) - v/c)*cos(theta2)/(1 - v*cos(theta2)/c))*(cos(theta2) - v/c)/((1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c)))/(c**2*sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: (1 - (np.cos(theta2) - v/c)*np.cos(theta2)/(1 - v*np.cos(theta2)/c))*(2*np.cos(theta2) - (1 - (np.cos(theta2) - v/c)*np.cos(theta2)/(1 - v*np.cos(theta2)/c))*(np.cos(theta2) - v/c)/((1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)))/(c**2*np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)**2) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-(-sin(theta2)/(1 - v*cos(theta2)/c) - v*(cos(theta2) - v/c)*sin(theta2)/(c*(1 - v*cos(theta2)/c)**2))/sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: -(-np.sin(theta2)/(1 - v*np.cos(theta2)/c) - v*(np.cos(theta2) - v/c)*np.sin(theta2)/(c*(1 - v*np.cos(theta2)/c)**2))/np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2) )(*args)'
      },
      {
        'name': 'theta2',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '(cos(theta2) - (1 + v*(cos(theta2) - v/c)/(c*(1 - v*cos(theta2)/c)))**2*(cos(theta2) - v/c)*sin(theta2)**2/((1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c)**2) + v*(cos(theta2) - v/c)*cos(theta2)/(c*(1 - v*cos(theta2)/c)) - 2*v*sin(theta2)**2/(c*(1 - v*cos(theta2)/c)) - 2*v**2*(cos(theta2) - v/c)*sin(theta2)**2/(c**2*(1 - v*cos(theta2)/c)**2))/(sqrt(1 - (cos(theta2) - v/c)**2/(1 - v*cos(theta2)/c)**2)*(1 - v*cos(theta2)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,theta2: (np.cos(theta2) - (1 + v*(np.cos(theta2) - v/c)/(c*(1 - v*np.cos(theta2)/c)))**2*(np.cos(theta2) - v/c)*np.sin(theta2)**2/((1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)**2) + v*(np.cos(theta2) - v/c)*np.cos(theta2)/(c*(1 - v*np.cos(theta2)/c)) - 2*v*np.sin(theta2)**2/(c*(1 - v*np.cos(theta2)/c)) - 2*v**2*(np.cos(theta2) - v/c)*np.sin(theta2)**2/(c**2*(1 - v*np.cos(theta2)/c)**2))/(np.sqrt(1 - (np.cos(theta2) - v/c)**2/(1 - v*np.cos(theta2)/c)**2)*(1 - v*np.cos(theta2)/c)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'c',
        'low': 4.0,
        'high': 6.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'theta2',
        'low': 1.0,
        'high': 3.0
      }
    ]
  },
  {
    'EquationName': 'Bonus11',
    'DescriptiveName': 'Bonus11.0, N-slit diffraction',
    'Constraints': [
      {
        'name': 'I_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '4*sin(alpha/2)**2*sin(delta*n/2)**2/(alpha**2*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 4*np.sin(alpha/2)**2*np.sin(delta*n/2)**2/(alpha**2*np.sin(delta/2)**2) )(*args)'
      },
      {
        'name': 'I_0',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 0 )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '4*I_0*sin(alpha/2)*sin(delta*n/2)**2*cos(alpha/2)/(alpha**2*sin(delta/2)**2) - 8*I_0*sin(alpha/2)**2*sin(delta*n/2)**2/(alpha**3*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 4*I_0*np.sin(alpha/2)*np.sin(delta*n/2)**2*np.cos(alpha/2)/(alpha**2*np.sin(delta/2)**2) - 8*I_0*np.sin(alpha/2)**2*np.sin(delta*n/2)**2/(alpha**3*np.sin(delta/2)**2) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*I_0*(-sin(alpha/2)**2 + cos(alpha/2)**2 - 8*sin(alpha/2)*cos(alpha/2)/alpha + 12*sin(alpha/2)**2/alpha**2)*sin(delta*n/2)**2/(alpha**2*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 2*I_0*(-np.sin(alpha/2)**2 + np.cos(alpha/2)**2 - 8*np.sin(alpha/2)*np.cos(alpha/2)/alpha + 12*np.sin(alpha/2)**2/alpha**2)*np.sin(delta*n/2)**2/(alpha**2*np.sin(delta/2)**2) )(*args)'
      },
      {
        'name': 'delta',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '4*I_0*n*sin(alpha/2)**2*sin(delta*n/2)*cos(delta*n/2)/(alpha**2*sin(delta/2)**2) - 4*I_0*sin(alpha/2)**2*sin(delta*n/2)**2*cos(delta/2)/(alpha**2*sin(delta/2)**3)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 4*I_0*n*np.sin(alpha/2)**2*np.sin(delta*n/2)*np.cos(delta*n/2)/(alpha**2*np.sin(delta/2)**2) - 4*I_0*np.sin(alpha/2)**2*np.sin(delta*n/2)**2*np.cos(delta/2)/(alpha**2*np.sin(delta/2)**3) )(*args)'
      },
      {
        'name': 'delta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*I_0*(-n**2*(sin(delta*n/2)**2 - cos(delta*n/2)**2) - 4*n*sin(delta*n/2)*cos(delta/2)*cos(delta*n/2)/sin(delta/2) + (1 + 3*cos(delta/2)**2/sin(delta/2)**2)*sin(delta*n/2)**2)*sin(alpha/2)**2/(alpha**2*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 2*I_0*(-n**2*(np.sin(delta*n/2)**2 - np.cos(delta*n/2)**2) - 4*n*np.sin(delta*n/2)*np.cos(delta/2)*np.cos(delta*n/2)/np.sin(delta/2) + (1 + 3*np.cos(delta/2)**2/np.sin(delta/2)**2)*np.sin(delta*n/2)**2)*np.sin(alpha/2)**2/(alpha**2*np.sin(delta/2)**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '4*I_0*delta*sin(alpha/2)**2*sin(delta*n/2)*cos(delta*n/2)/(alpha**2*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: 4*I_0*delta*np.sin(alpha/2)**2*np.sin(delta*n/2)*np.cos(delta*n/2)/(alpha**2*np.sin(delta/2)**2) )(*args)'
      },
      {
        'name': 'n',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-2*I_0*delta**2*(sin(delta*n/2)**2 - cos(delta*n/2)**2)*sin(alpha/2)**2/(alpha**2*sin(delta/2)**2)',
        'derivative_lambda': 'lambda args : (lambda I_0,alpha,delta,n: -2*I_0*delta**2*(np.sin(delta*n/2)**2 - np.cos(delta*n/2)**2)*np.sin(alpha/2)**2/(alpha**2*np.sin(delta/2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'I_0',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'delta',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'n',
        'low': 1.0,
        'high': 2.0
      }
    ]
  },
  {
    'EquationName': 'Bonus12',
    'DescriptiveName': 'Bonus12.0, 2.11 Jackson',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-d*q*y/(4*pi*epsilon*(-d**2 + y**2)**2) + (4*pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(4*pi*epsilon*y**2)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: -d*q*y/(4*np.pi*epsilon*(-d**2 + y**2)**2) + (4*np.pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(4*np.pi*epsilon*y**2) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-d*y/(2*pi*epsilon*(d**2 - y**2)**2)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: -d*y/(2*np.pi*epsilon*(d**2 - y**2)**2) )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'q*(4*d*q*y**4/(-d**2 + y**2)**3 - 3*d*q*y**2/(-d**2 + y**2)**2)/(4*pi*epsilon*y**2) - q*(4*pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(2*pi*epsilon*y**3)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: q*(4*d*q*y**4/(-d**2 + y**2)**3 - 3*d*q*y**2/(-d**2 + y**2)**2)/(4*np.pi*epsilon*y**2) - q*(4*np.pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(2*np.pi*epsilon*y**3) )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'd*q*(q*(4*y**2/(d**2 - y**2) + 3)/(d**2 - y**2)**2 - q*(12*y**4/(d**2 - y**2)**2 + 14*y**2/(d**2 - y**2) + 3)/(2*(d**2 - y**2)**2) + 3*(4*pi*Volt*epsilon - q*y**3/(d**2 - y**2)**2)/(2*y**3))/(pi*epsilon*y)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: d*q*(q*(4*y**2/(d**2 - y**2) + 3)/(d**2 - y**2)**2 - q*(12*y**4/(d**2 - y**2)**2 + 14*y**2/(d**2 - y**2) + 3)/(2*(d**2 - y**2)**2) + 3*(4*np.pi*Volt*epsilon - q*y**3/(d**2 - y**2)**2)/(2*y**3))/(np.pi*epsilon*y) )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'd*q/y**2',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: d*q/y**2 )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: 0 )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*(4*pi*Volt*epsilon - 4*d**2*q*y**3/(-d**2 + y**2)**3 - q*y**3/(-d**2 + y**2)**2)/(4*pi*epsilon*y**2)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: q*(4*np.pi*Volt*epsilon - 4*d**2*q*y**3/(-d**2 + y**2)**3 - q*y**3/(-d**2 + y**2)**2)/(4*np.pi*epsilon*y**2) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*d*q**2*y*(2*d**2/(d**2 - y**2) - 1)/(pi*epsilon*(d**2 - y**2)**3)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: -3*d*q**2*y*(2*d**2/(d**2 - y**2) - 1)/(np.pi*epsilon*(d**2 - y**2)**3) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'Volt*d*q/(epsilon*y**2) - q*(4*pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(4*pi*epsilon**2*y**2)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: Volt*d*q/(epsilon*y**2) - q*(4*np.pi*Volt*d*epsilon - d*q*y**3/(-d**2 + y**2)**2)/(4*np.pi*epsilon**2*y**2) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'd*q*(-2*Volt + (4*pi*Volt*epsilon - q*y**3/(d**2 - y**2)**2)/(2*pi*epsilon))/(epsilon**2*y**2)',
        'derivative_lambda': 'lambda args : (lambda q,y,Volt,d,epsilon: d*q*(-2*Volt + (4*np.pi*Volt*epsilon - q*y**3/(d**2 - y**2)**2)/(2*np.pi*epsilon))/(epsilon**2*y**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'Volt',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 4.0,
        'high': 6.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus13',
    'DescriptiveName': 'Bonus13.0, 3.45 Jackson',
    'Constraints': [
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/(4*pi*epsilon*sqrt(d**2 - 2*d*r*cos(alpha) + r**2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: 1/(4*np.pi*epsilon*np.sqrt(d**2 - 2*d*r*np.cos(alpha) + r**2)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'q*(d*cos(alpha) - r)/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: q*(d*np.cos(alpha) - r)/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'q*(3*(d*cos(alpha) - r)**2/(d**2 - 2*d*r*cos(alpha) + r**2) - 1)/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: q*(3*(d*np.cos(alpha) - r)**2/(d**2 - 2*d*r*np.cos(alpha) + r**2) - 1)/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'q*(-d + r*cos(alpha))/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: q*(-d + r*np.cos(alpha))/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q*(3*(d - r*cos(alpha))**2/(d**2 - 2*d*r*cos(alpha) + r**2) - 1)/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: q*(3*(d - r*np.cos(alpha))**2/(d**2 - 2*d*r*np.cos(alpha) + r**2) - 1)/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-d*q*r*sin(alpha)/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: -d*q*r*np.sin(alpha)/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'd*q*r*(3*d*r*sin(alpha)**2/(d**2 - 2*d*r*cos(alpha) + r**2) - cos(alpha))/(4*pi*epsilon*(d**2 - 2*d*r*cos(alpha) + r**2)**(3/2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: d*q*r*(3*d*r*np.sin(alpha)**2/(d**2 - 2*d*r*np.cos(alpha) + r**2) - np.cos(alpha))/(4*np.pi*epsilon*(d**2 - 2*d*r*np.cos(alpha) + r**2)**(3/2)) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-q/(4*pi*epsilon**2*sqrt(d**2 - 2*d*r*cos(alpha) + r**2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: -q/(4*np.pi*epsilon**2*np.sqrt(d**2 - 2*d*r*np.cos(alpha) + r**2)) )(*args)'
      },
      {
        'name': 'epsilon',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q/(2*pi*epsilon**3*sqrt(d**2 - 2*d*r*cos(alpha) + r**2))',
        'derivative_lambda': 'lambda args : (lambda q,r,d,alpha,epsilon: q/(2*np.pi*epsilon**3*np.sqrt(d**2 - 2*d*r*np.cos(alpha) + r**2)) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'd',
        'low': 4.0,
        'high': 6.0
      },
      {
        'name': 'alpha',
        'low': 0.0,
        'high': 6.0
      },
      {
        'name': 'epsilon',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus14',
    'DescriptiveName': "Bonus14.0, 4.60' Jackson",
    'Constraints': [
      {
        'name': 'Ef',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: (d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*np.cos(theta) )(*args)'
      },
      {
        'name': 'Ef',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-Ef*(d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*sin(theta)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: -Ef*(d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*np.sin(theta) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '-Ef*(d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: -Ef*(d**3*(alpha - 1)/(r**2*(alpha + 2)) - r)*np.cos(theta) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Ef*(-2*d**3*(alpha - 1)/(r**3*(alpha + 2)) - 1)*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: Ef*(-2*d**3*(alpha - 1)/(r**3*(alpha + 2)) - 1)*np.cos(theta) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '6*Ef*d**3*(alpha - 1)*cos(theta)/(r**4*(alpha + 2))',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: 6*Ef*d**3*(alpha - 1)*np.cos(theta)/(r**4*(alpha + 2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '3*Ef*d**2*(alpha - 1)*cos(theta)/(r**2*(alpha + 2))',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: 3*Ef*d**2*(alpha - 1)*np.cos(theta)/(r**2*(alpha + 2)) )(*args)'
      },
      {
        'name': 'd',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '6*Ef*d*(alpha - 1)*cos(theta)/(r**2*(alpha + 2))',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: 6*Ef*d*(alpha - 1)*np.cos(theta)/(r**2*(alpha + 2)) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'Ef*(-d**3*(alpha - 1)/(r**2*(alpha + 2)**2) + d**3/(r**2*(alpha + 2)))*cos(theta)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: Ef*(-d**3*(alpha - 1)/(r**2*(alpha + 2)**2) + d**3/(r**2*(alpha + 2)))*np.cos(theta) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '2*Ef*d**3*((alpha - 1)/(alpha + 2) - 1)*cos(theta)/(r**2*(alpha + 2)**2)',
        'derivative_lambda': 'lambda args : (lambda Ef,theta,r,d,alpha: 2*Ef*d**3*((alpha - 1)/(alpha + 2) - 1)*np.cos(theta)/(r**2*(alpha + 2)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'Ef',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 0.0,
        'high': 6.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'd',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus15',
    'DescriptiveName': 'Bonus15.0, 11.38 Jackson',
    'Constraints': [
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'omega*v*sqrt(1 - v**2/c**2)*cos(theta)/(c**2*(1 + v*cos(theta)/c)**2) + omega*v**2/(c**3*sqrt(1 - v**2/c**2)*(1 + v*cos(theta)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: omega*v*np.sqrt(1 - v**2/c**2)*np.cos(theta)/(c**2*(1 + v*np.cos(theta)/c)**2) + omega*v**2/(c**3*np.sqrt(1 - v**2/c**2)*(1 + v*np.cos(theta)/c)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'omega*v*(-2*sqrt(1 - v**2/c**2)*(1 - v*cos(theta)/(c*(1 + v*cos(theta)/c)))*cos(theta)/(1 + v*cos(theta)/c) - v*(3 + v**2/(c**2*(1 - v**2/c**2)))/(c*sqrt(1 - v**2/c**2)) + 2*v**2*cos(theta)/(c**2*sqrt(1 - v**2/c**2)*(1 + v*cos(theta)/c)))/(c**3*(1 + v*cos(theta)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: omega*v*(-2*np.sqrt(1 - v**2/c**2)*(1 - v*np.cos(theta)/(c*(1 + v*np.cos(theta)/c)))*np.cos(theta)/(1 + v*np.cos(theta)/c) - v*(3 + v**2/(c**2*(1 - v**2/c**2)))/(c*np.sqrt(1 - v**2/c**2)) + 2*v**2*np.cos(theta)/(c**2*np.sqrt(1 - v**2/c**2)*(1 + v*np.cos(theta)/c)))/(c**3*(1 + v*np.cos(theta)/c)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-omega*sqrt(1 - v**2/c**2)*cos(theta)/(c*(1 + v*cos(theta)/c)**2) - omega*v/(c**2*sqrt(1 - v**2/c**2)*(1 + v*cos(theta)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: -omega*np.sqrt(1 - v**2/c**2)*np.cos(theta)/(c*(1 + v*np.cos(theta)/c)**2) - omega*v/(c**2*np.sqrt(1 - v**2/c**2)*(1 + v*np.cos(theta)/c)) )(*args)'
      },
      {
        'name': 'v',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'omega*(2*sqrt(1 - v**2/c**2)*cos(theta)**2/(1 + v*cos(theta)/c)**2 - (1 + v**2/(c**2*(1 - v**2/c**2)))/sqrt(1 - v**2/c**2) + 2*v*cos(theta)/(c*sqrt(1 - v**2/c**2)*(1 + v*cos(theta)/c)))/(c**2*(1 + v*cos(theta)/c))',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: omega*(2*np.sqrt(1 - v**2/c**2)*np.cos(theta)**2/(1 + v*np.cos(theta)/c)**2 - (1 + v**2/(c**2*(1 - v**2/c**2)))/np.sqrt(1 - v**2/c**2) + 2*v*np.cos(theta)/(c*np.sqrt(1 - v**2/c**2)*(1 + v*np.cos(theta)/c)))/(c**2*(1 + v*np.cos(theta)/c)) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'sqrt(1 - v**2/c**2)/(1 + v*cos(theta)/c)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: np.sqrt(1 - v**2/c**2)/(1 + v*np.cos(theta)/c) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: 0 )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'omega*v*sqrt(1 - v**2/c**2)*sin(theta)/(c*(1 + v*cos(theta)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: omega*v*np.sqrt(1 - v**2/c**2)*np.sin(theta)/(c*(1 + v*np.cos(theta)/c)**2) )(*args)'
      },
      {
        'name': 'theta',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'omega*v*sqrt(1 - v**2/c**2)*(cos(theta) + 2*v*sin(theta)**2/(c*(1 + v*cos(theta)/c)))/(c*(1 + v*cos(theta)/c)**2)',
        'derivative_lambda': 'lambda args : (lambda c,v,omega,theta: omega*v*np.sqrt(1 - v**2/c**2)*(np.cos(theta) + 2*v*np.sin(theta)**2/(c*(1 + v*np.cos(theta)/c)))/(c*(1 + v*np.cos(theta)/c)**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'c',
        'low': 5.0,
        'high': 20.0
      },
      {
        'name': 'v',
        'low': 1.0,
        'high': 3.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'theta',
        'low': 0.0,
        'high': 6.0
      }
    ]
  },
  {
    'EquationName': 'Bonus16',
    'DescriptiveName': 'Bonus16.0, 8.56 Goldstein',
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**4*m/sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2)',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: c**4*m/np.sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**4*(-c**2*m**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: c**4*(-c**2*m**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/np.sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2)) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(2*c**3*m**2 + c*(-A_vec*q + p)**2)/sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2)',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: (2*c**3*m**2 + c*(-A_vec*q + p)**2)/np.sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(6*c**2*m**2 + (A_vec*q - p)**2 - (2*c**2*m**2 + (A_vec*q - p)**2)**2/(c**2*m**2 + (A_vec*q - p)**2))/sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: (6*c**2*m**2 + (A_vec*q - p)**2 - (2*c**2*m**2 + (A_vec*q - p)**2)**2/(c**2*m**2 + (A_vec*q - p)**2))/np.sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2)) )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'c**2*(-2*A_vec*q + 2*p)/(2*sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: c**2*(-2*A_vec*q + 2*p)/(2*np.sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2)) )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: c**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/np.sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2)) )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-A_vec*c**2*(-A_vec*q + p)/sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2) + Volt',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: -A_vec*c**2*(-A_vec*q + p)/np.sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2) + Volt )(*args)'
      },
      {
        'name': 'q',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'A_vec**2*c**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: A_vec**2*c**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/np.sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2)) )(*args)'
      },
      {
        'name': 'A_vec',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-c**2*q*(-A_vec*q + p)/sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2)',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: -c**2*q*(-A_vec*q + p)/np.sqrt(c**4*m**2 + c**2*(-A_vec*q + p)**2) )(*args)'
      },
      {
        'name': 'A_vec',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**2*q**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2))',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: c**2*q**2*(-(A_vec*q - p)**2/(c**2*m**2 + (A_vec*q - p)**2) + 1)/np.sqrt(c**2*(c**2*m**2 + (A_vec*q - p)**2)) )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'q',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: q )(*args)'
      },
      {
        'name': 'Volt',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,c,p,q,A_vec,Volt: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'p',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'q',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'A_vec',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'Volt',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus17',
    'DescriptiveName': "Bonus17.0, 12.80' Goldstein",
    'Constraints': [
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': 'omega**2*x**2*(alpha*x/y + 1) - (m**2*omega**2*x**2*(alpha*x/y + 1) + p**2)/(2*m**2)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: omega**2*x**2*(alpha*x/y + 1) - (m**2*omega**2*x**2*(alpha*x/y + 1) + p**2)/(2*m**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(-omega**2*x**2*(alpha*x/y + 1) + (m**2*omega**2*x**2*(alpha*x/y + 1) + p**2)/m**2)/m',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: (-omega**2*x**2*(alpha*x/y + 1) + (m**2*omega**2*x**2*(alpha*x/y + 1) + p**2)/m**2)/m )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*omega*x**2*(alpha*x/y + 1)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: m*omega*x**2*(alpha*x/y + 1) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*x**2*(alpha*x/y + 1)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: m*x**2*(alpha*x/y + 1) )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'p/m',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: p/m )(*args)'
      },
      {
        'name': 'p',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '1/m',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: 1/m )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha*m*omega**2*x**3/(2*y**2)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: -alpha*m*omega**2*x**3/(2*y**2) )(*args)'
      },
      {
        'name': 'y',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*m*omega**2*x**3/y**3',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: alpha*m*omega**2*x**3/y**3 )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '(alpha*m**2*omega**2*x**2/y + 2*m**2*omega**2*x*(alpha*x/y + 1))/(2*m)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: (alpha*m**2*omega**2*x**2/y + 2*m**2*omega**2*x*(alpha*x/y + 1))/(2*m) )(*args)'
      },
      {
        'name': 'x',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*omega**2*(3*alpha*x/y + 1)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: m*omega**2*(3*alpha*x/y + 1) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'm*omega**2*x**3/(2*y)',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: m*omega**2*x**3/(2*y) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda m,omega,p,y,x,alpha: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'p',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'y',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'x',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus18',
    'DescriptiveName': 'Bonus18.0, 15.2.1 Weinberg',
    'Constraints': [
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*(H_G**2 + c**2*k_f/r**2)/(8*pi*G**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: -3*(H_G**2 + c**2*k_f/r**2)/(8*np.pi*G**2) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*(H_G**2 + c**2*k_f/r**2)/(4*pi*G**3)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3*(H_G**2 + c**2*k_f/r**2)/(4*np.pi*G**3) )(*args)'
      },
      {
        'name': 'k_f',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*c**2/(8*pi*G*r**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3*c**2/(8*np.pi*G*r**2) )(*args)'
      },
      {
        'name': 'k_f',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*c**2*k_f/(4*pi*G*r**3)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: -3*c**2*k_f/(4*np.pi*G*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '9*c**2*k_f/(4*pi*G*r**4)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 9*c**2*k_f/(4*np.pi*G*r**4) )(*args)'
      },
      {
        'name': 'H_G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*H_G/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3*H_G/(4*np.pi*G) )(*args)'
      },
      {
        'name': 'H_G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3/(4*np.pi*G) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*c*k_f/(4*pi*G*r**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3*c*k_f/(4*np.pi*G*r**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*k_f/(4*pi*G*r**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,c: 3*k_f/(4*np.pi*G*r**2) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'G',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'k_f',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'H_G',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus19',
    'DescriptiveName': 'Bonus19.0, 15.2.2 Weinberg',
    'Constraints': [
      {
        'name': 'G',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '(H_G**2*c**2*(1 - 2*alpha) + c**4*k_f/r**2)/(8*pi*G**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: (H_G**2*c**2*(1 - 2*alpha) + c**4*k_f/r**2)/(8*np.pi*G**2) )(*args)'
      },
      {
        'name': 'G',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'c**2*(H_G**2*(2*alpha - 1) - c**2*k_f/r**2)/(4*pi*G**3)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: c**2*(H_G**2*(2*alpha - 1) - c**2*k_f/r**2)/(4*np.pi*G**3) )(*args)'
      },
      {
        'name': 'k_f',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-c**4/(8*pi*G*r**2)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: -c**4/(8*np.pi*G*r**2) )(*args)'
      },
      {
        'name': 'k_f',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: 0 )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**4*k_f/(4*pi*G*r**3)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: c**4*k_f/(4*np.pi*G*r**3) )(*args)'
      },
      {
        'name': 'r',
        'order_derivative': 2,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-3*c**4*k_f/(4*pi*G*r**4)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: -3*c**4*k_f/(4*np.pi*G*r**4) )(*args)'
      },
      {
        'name': 'H_G',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': '-H_G*c**2*(1 - 2*alpha)/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: -H_G*c**2*(1 - 2*alpha)/(4*np.pi*G) )(*args)'
      },
      {
        'name': 'H_G',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'c**2*(2*alpha - 1)/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: c**2*(2*alpha - 1)/(4*np.pi*G) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'H_G**2*c**2/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: H_G**2*c**2/(4*np.pi*G) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: 0 )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'None',
        'derivative': '-(2*H_G**2*c*(1 - 2*alpha) + 4*c**3*k_f/r**2)/(8*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: -(2*H_G**2*c*(1 - 2*alpha) + 4*c**3*k_f/r**2)/(8*np.pi*G) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': '(H_G**2*(2*alpha - 1) - 6*c**2*k_f/r**2)/(4*pi*G)',
        'derivative_lambda': 'lambda args : (lambda G,k_f,r,H_G,alpha,c: (H_G**2*(2*alpha - 1) - 6*c**2*k_f/r**2)/(4*np.pi*G) )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'G',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'k_f',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'r',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'H_G',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      }
    ]
  },
  {
    'EquationName': 'Bonus20',
    'DescriptiveName': 'Bonus20.0, Klein-Nishina (13.132 Schwarz)',
    'Constraints': [
      {
        'name': 'omega',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': 'alpha**2*h**2*omega_0**2*(1/omega_0 - omega_0/omega**2)/(4*pi*c**2*m**2*omega**2) - alpha**2*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**3)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*h**2*omega_0**2*(1/omega_0 - omega_0/omega**2)/(4*np.pi*c**2*m**2*omega**2) - alpha**2*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**3) )(*args)'
      },
      {
        'name': 'omega',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha**2*h**2*omega_0**2*(-1/omega_0 + 3*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*omega) + 3*omega_0/(2*omega**2))/(pi*c**2*m**2*omega**3)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*h**2*omega_0**2*(-1/omega_0 + 3*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*omega) + 3*omega_0/(2*omega**2))/(np.pi*c**2*m**2*omega**3) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha**2*h**2*omega_0**2*(-omega/omega_0**2 + 1/omega)/(4*pi*c**2*m**2*omega**2) + alpha**2*h**2*omega_0*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*h**2*omega_0**2*(-omega/omega_0**2 + 1/omega)/(4*np.pi*c**2*m**2*omega**2) + alpha**2*h**2*omega_0*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**2) )(*args)'
      },
      {
        'name': 'omega_0',
        'order_derivative': 2,
        'descriptor': 'None',
        'derivative': 'alpha**2*h**2*(1/(2*omega_0) - omega_0*(omega/omega_0**2 - 1/omega)/omega + (omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*omega))/(pi*c**2*m**2*omega)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*h**2*(1/(2*omega_0) - omega_0*(omega/omega_0**2 - 1/omega)/omega + (omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*omega))/(np.pi*c**2*m**2*omega) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**2) )(*args)'
      },
      {
        'name': 'alpha',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 1,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha**2*h*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*h*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**2) )(*args)'
      },
      {
        'name': 'h',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': 'alpha**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: alpha**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**2*omega**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha**2*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**3*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: -alpha**2*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**3*omega**2) )(*args)'
      },
      {
        'name': 'm',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*alpha**2*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**2*m**4*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 3*alpha**2*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**2*m**4*omega**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 1,
        'descriptor': 'strong monotonic decreasing',
        'derivative': '-alpha**2*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**3*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: -alpha**2*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**3*m**2*omega**2) )(*args)'
      },
      {
        'name': 'c',
        'order_derivative': 2,
        'descriptor': 'strong monotonic increasing',
        'derivative': '3*alpha**2*h**2*omega_0**2*(omega/omega_0 - sin(beta)**2 + omega_0/omega)/(2*pi*c**4*m**2*omega**2)',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 3*alpha**2*h**2*omega_0**2*(omega/omega_0 - np.sin(beta)**2 + omega_0/omega)/(2*np.pi*c**4*m**2*omega**2) )(*args)'
      },
      {
        'name': 'beta',
        'order_derivative': 1,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 0 )(*args)'
      },
      {
        'name': 'beta',
        'order_derivative': 2,
        'descriptor': 'constant',
        'derivative': '0',
        'derivative_lambda': 'lambda args : (lambda omega,omega_0,alpha,h,m,c,beta: 0 )(*args)'
      }
    ],
    'Variables': [
      {
        'name': 'omega',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'omega_0',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'alpha',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'h',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'm',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'c',
        'low': 1.0,
        'high': 5.0
      },
      {
        'name': 'beta',
        'low': 0.0,
        'high': 6.0
      }
    ]
  },
]
