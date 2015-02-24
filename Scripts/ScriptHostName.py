
import socket
LocalHost=socket.gethostname()
if "." in LocalHost: LocalHost,_=LocalHost.split(".")
LocalHostName=LocalHost
