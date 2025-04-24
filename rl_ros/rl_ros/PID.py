"""
Baseline PID controller

Note Include a setpoint filter 
Setpoint Filter - Using a First Order Lag filter
https://blog.opticontrols.com/archives/1319#:~:text=A%20first%2Dorder%20lag%20filter%20is%20a%20type,more%20to%20the%20output%20than%20older%20samples**
https://cookierobotics.com/084/
"""

class FirstOrderFilter:
    """
    First-order filter class for smoothing a setpoint signal.
    https://en.wikipedia.org/wiki/Low-pass_filter
    Args:
        tau (float): Time constant of the filter.
        dt (float): Time step for the filter.
        x0 (float): Initial value of the filter.
    Methods:
        filter(x: float) -> float:
            Applies the first-order filter to the input signal.
    """
    def __init__(self, tau:float, dt:float, 
                 x0:float) -> None:
        self.tau:float = tau
        self.dt:float = dt
        self.x0:float = x0
        self.alpha:float = dt / (tau + dt)
        
    def filter(self, x:float) -> float:
        """
        Applies the first-order filter to the input signal.
        Args:
            x (float): Input signal to be filtered.
        Returns:
            float: Filtered output signal.
        """
        self.x0 = (1 - self.alpha) * self.x0 + self.alpha * x
        return self.x0

class PID:
    """
    PID controller class for controlling a system with a setpoint and current value.
    Args:
        min_constraint (float): Minimum constraint for the output.
        max_constraint (float): Maximum constraint for the output.
        use_integral (bool): Flag to use integral term in PID control.
        use_derivative (bool): Flag to use derivative term in PID control.
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step for the controller.
    
    Methods:
        compute(setpoint: float, current_value: float, dt: float) -> float:
            Computes the PID control output based on the setpoint and current value.
            
    """
    def __init__(self,
        min_constraint:float,
        max_constraint:float,
        use_integral:bool = False,
        use_derivative:bool = False,
        kp:float=0.05,
        ki:float=0.0,
        kd:float=0.0,
        dt:float=0.05) -> None:

        self.min_constraint:float = min_constraint
        self.max_constraint:float = max_constraint
        self.dt:float = dt
                
        self.use_integral:bool = use_integral
        self.use_derivative:bool = use_derivative
        
        self.kp:float = kp
        self.ki:float = ki
        self.kd:float = kd
        self.prev_error: float = None
        self.integral: float = 0.0
        
    def compute(self,
        setpoint:float,
        current_value:float,
        dt:float) -> float:
        
        error:float = setpoint - current_value
        derivative:float = (error - self.prev_error) / dt
        self.integral += error * dt
        
        if self.use_integral and self.use_derivative:
            output = (self.kp * error) + \
                (self.ki * self.integral) + (self.kd * derivative)
        elif self.use_integral:
            output:float = (self.kp * error) + \
                (self.ki * self.integral)
        elif self.use_derivative:
            output:float = (self.kp * error) + (self.kd * derivative)
        else:
            output:float = (self.kp * error)
        
        self.prev_error = error
        
        return output
    
    