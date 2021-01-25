These are install instructions specific for CentOS.

On a new VM, these packages need to be installed:
```
     sudo yum install -y python3
     sudo yum install -y python3-devel
     sudo yum install -y git
     sudo yum install -y gcc
     sudo yum install -y gcc-c++
```

Bazel is a little bit more complicated. Download the file found here:
```
  https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
```

to this directory as `bazel3.repo` (the file should only contain ~10 lines):
```
    /etc/yum.repos.d/
```

and run:
```
    sudo yum install -y bazel3
```  

All other instructions from the main `README.md` still apply.
