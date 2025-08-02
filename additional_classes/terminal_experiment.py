class TerminalExperiment:
    def __init__(self, log_on_screen=True):
        self.log_on_screen = log_on_screen
        pass

    def log(self, method_name, *args, **kwargs):
        if not self.log_on_screen:
            return
        # if kwargs exist, print them, otherwise do not
        if kwargs:
            log_message = f"Method: {method_name}, Args: {args}, Kwargs: {kwargs}"
        else:
            log_message = f"{method_name}: {args}"
        print(log_message)

    def __getattr__(self, method_name):
        def log_and_call(*args, **kwargs):
            self.log(method_name, *args, **kwargs)

        return log_and_call