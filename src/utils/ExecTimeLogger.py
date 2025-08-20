from functools import wraps
import logging
import os
import time


class ExecTimeLogger:
    """Logger for execution times of Python methods."""

    def __init__(self):
        """
        Initialize logger object and create log file if `LogFolder` set as environment variable.
        To create log folder as environment variable use the code below:

        Examples
        --------
        >>> import os
        >>> os.environ["LogFolder"] = 'LogFolder'
        >>> import ExecTimeLogger
        >>> exec_time_logger = ExecTimeLogger()
        >>> @exec_time_logger.exec_time
        >>> def func():
        """

        self.log_folder = os.environ.get('LogFolder', False)

        if self.log_folder:
            # Create logger object and configure it
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel('DEBUG')
            formatter = logging.Formatter('{asctime}, {levelname}, {message}',
                                          style='{', datefmt="%Y-%m-%d %H:%M:%S", )

            # Create log file with headers
            file_path = self.__create_log_file(self.log_folder)

            # Create file handler and configure it
            file_handler = logging.FileHandler(file_path, mode='a', encoding="utf-8")
            file_handler.setLevel('DEBUG')
            file_handler.setFormatter(formatter)

            # Add file handler to logger
            self.logger.addHandler(file_handler)

    def __create_log_file(self, file_folder):
        """Create a log file with header information if it does not exist.

        Parameters
        ----------
        file_folder : str
            The directory where the log file will be stored.

        Returns
        -------
        str
            The path to the log file.
        """

        file_name = 'log.csv'
        file_path = os.path.join(file_folder, file_name)

        if not os.path.isfile(file_path):
            header = 'Timestap,LogLevel,MethodName,ExecutionTime(inSecond)\n'
            with open(file_name, 'w') as writer:
                writer.writelines(header)
        else:
            with open(file_name, 'a') as writer:
                writer.writelines('\n')

        return file_path

    def exec_time(self, func):
        """Decorator to log the execution time of the given method.

        Parameters
        ----------
        func : callable
            The function whose execution time is to be logged.

        Returns
        -------
        callable
            A wrapped function that logs its execution time.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            output = func(*args, **kwargs)
            if self.log_folder:
                exec_time = round(time.time() - start_time, 5)
                self.logger.debug("{}, {}".format(func.__qualname__, exec_time))
            return output

        return wrapper