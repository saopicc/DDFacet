#!/usr/bin/env python3
import os
import time


import numpy as np
import psutil
import pylab
import socket
import glob
import sys
import os
#pylab.ion()
import optparse
import pickle

SaveFile="last_IOMonitor.obj"

def read_options():
    desc=""" """
    opt = optparse.OptionParser(usage='Task to start a monitoring task, Usage: %prog <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* General options")
    group.add_option('--Path',type=str,help="cwd",default="PWD")
    group.add_option('--Interval',type=float,help="interval",default=1.)
    group.add_option('--Mode',type=str,help="Print/Dump",default="Dump")
    group.add_option('--Reset',type=int,help="Reset dump",default=1)
    

    opt.add_option_group(group)

    
    options, arguments = opt.parse_args()

    f = open(SaveFile,"wb")
    pickle.dump(options,f)


def get_sector_size(dev):
    path = f"/sys/class/block/{dev}/queue/logical_block_size"
    with open(path) as f:
        return int(f.read().strip())

# ------------------------------------------------------------
# Step 1: find mount point + source device for a path
# ------------------------------------------------------------
def find_mount(path):
    path = os.path.realpath(path)

    best = None
    with open("/proc/self/mountinfo") as f:
        for line in f:
            parts = line.split()
            mount_point = parts[4]
            source = parts[-2]

            if path == mount_point or path.startswith(mount_point + "/"):
                if best is None or len(mount_point) > len(best[0]):
                    best = (mount_point, source)

    if best is None:
        raise RuntimeError(f"No mount point found for {path}")

    return best  # (mount_point, source)


# ------------------------------------------------------------
# Step 2: map /dev/... → diskstats device name (dm-X, sda, nvme...)
# ------------------------------------------------------------
def dev_to_diskstats_name(dev):
    if not dev.startswith("/dev/"):
        raise RuntimeError(f"{dev} is not a local block device")
    real = os.path.realpath(dev)
    name = os.path.basename(real)

    # LVM devices
    if name.startswith("dm-"):
        return name

    # Normal disks / partitions
    if os.path.exists(f"/sys/class/block/{name}"):
        return name

    raise RuntimeError(f"Cannot resolve diskstats device for {dev}")


# ------------------------------------------------------------
# Step 3: read diskstats counters
# ------------------------------------------------------------
def read_diskstats(dev):
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if parts[2] == dev:
                sectors_read = int(parts[5])
                sectors_written = int(parts[9])
                return sectors_read, sectors_written
    raise RuntimeError(f"{dev} not found in /proc/diskstats")


# ------------------------------------------------------------
# Step 4: public helper – path → device
# ------------------------------------------------------------
def diskstats_device_for_path(path):


    mount, source = find_mount(path)

    if not source.startswith("/dev/"):
        return None, mount  # network / tmpfs / fuse

    dev = dev_to_diskstats_name(source)
    
    mount_point, source = find_mount(path)
    dev = dev_to_diskstats_name(source)
    return dev, mount_point






# ------------------------------------------------------------
# Step 5: measure I/O for that path
# ------------------------------------------------------------
import os
import time

UID = os.getuid()

def get_user_pids_nfs():
    pids = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = os.stat(f"/proc/{pid}")
            if stat.st_uid == UID:
                pids.append(pid)
        except FileNotFoundError:
            pass  # process exited
    return pids


def read_proc_io_nfs(pid):
    io = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, v = line.split(":")
                io[k.strip()] = int(v.strip())
        return io
    except (FileNotFoundError, PermissionError):
        return None


def sample_processes_nfs():
    snapshot = {}
    for pid in get_user_pids_nfs():
        io = read_proc_io_nfs(pid)
        if io:
            snapshot[pid] = io
    return snapshot

def get_cmdline_nfs(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode().strip()
    except Exception:
        return ""



# ###########################################        

class ClassIO():
    def __init__(self,options=None):
        self.path=options.Path
        self.Mode=options.Mode
        self.Method="NFS"
        if self.path=="cwd":
            self.path=os.getcwd()
        self.interval=options.Interval
        self.HostName=socket.gethostname()
        self.FileDump=os.path.expanduser("~/DDF_io_monitor.%s.csv"%self.HostName)
        print("Logging to %s"%self.FileDump)
        if options.Reset:
            os.system("rm %s"%self.FileDump)

    def measure_io(self):
        path=self.path
        interval=self.interval
        dev, mount = diskstats_device_for_path(path)
        
        if dev is None:
            # nfs drive
            stop

        r1, w1 = read_diskstats(dev)
        t1 = time.time()
    
        time.sleep(interval)
    
        r2, w2 = read_diskstats(dev)
        t2 = time.time()
    
        dt = t2 - t1
        SECTOR_SIZE= get_sector_size(dev)
    
        read_MBps  = (r2 - r1) * SECTOR_SIZE / dt / 1024 / 1024
        write_MBps = (w2 - w1) * SECTOR_SIZE / dt / 1024 / 1024
        return read_MBps,write_MBps
        
    def measure_io_nfs(self):
        t1 = time.time()
        s1 = sample_processes_nfs()
        time.sleep(1)
        t2 = time.time()
        s2 = sample_processes_nfs()
        dt = t2 - t1
        results = []
        for pid in s1.keys() & s2.keys():
            rbytes = s2[pid]["read_bytes"] - s1[pid]["read_bytes"]
            wbytes = s2[pid]["write_bytes"] - s1[pid]["write_bytes"]
            if rbytes > 0 or wbytes > 0:
                results.append((
                    pid,
                    rbytes / dt,
                    wbytes / dt,
                    get_cmdline_nfs(pid)
                ))
        # Sort by total I/O
        results.sort(key=lambda x: x[1] + x[2], reverse=True)
        # ---- Display ----
        rtot=0.
        wtot=0.
        # print(f"{'PID':>6} {'READ KB/s':>12} {'WRITE KB/s':>12} COMMAND")
        for pid, r, w, cmd in results:
            rtot+=r
            wtot+=w
            # print(f"{pid:>6} {r/1024:>12.1f} {w/1024:>12.1f} {cmd[:80]}")
        return rtot/1024**2,wtot/1024**2
    
    def startMonitor(self):
        while True:
            AbsTime=time.time()
            if self.Mode=="Dump":
                read_MBps, write_MBps=self.measure_io_nfs()
                # if self.Method=="NFS":
                #     read_MBps, write_MBps=self.measure_io_nfs()
                # else:
                #     read_MBps, write_MBps=self.measure_io()
                with open(self.FileDump, 'a') as file:
                    s=f"{AbsTime}, {read_MBps}, {write_MBps}"
                    file.write('%s\n'%s)

        # return {
        #     "path": path,
        #     "mount_point": mount,
        #     "device": dev,
        #     "read_MBps": read_MBps,
        #     "write_MBps": write_MBps,
        # }
    


def driver():
    read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)

    MM=ClassIO(options=options)
    MM.startMonitor()
    
if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()

