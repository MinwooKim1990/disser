def F(integration_variables, real_parameters, complex_parameters):
    x1,x2,x3=integration_variables
    s12,s14,mh2=real_parameters
    expr=(+1+2*x3*1-x3*mh2+x3**2*1+2*x2*1-x2*s12+2*x2*x3*1-x2*x3*mh2+x2**2*1+2*x1*1+2*x1*x3*1-x1*x3*s14+2*x1*x2*1+x1**2*1)
    return expr

def U(integration_variables, real_parameters, complex_parameters):
    x1,x2,x3=integration_variables
    s12,s14,mh2=real_parameters
    expr=(+1+x3+x2+x1)
    return expr

def I0(integration_variables, real_parameters, complex_parameters):
    x1,x2,x3=integration_variables
    s12,s14,mh2=real_parameters
    expr=(2/(F(integration_variables, real_parameters, complex_parameters)**2))
    return expr

