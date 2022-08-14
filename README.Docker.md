# Get Started Using Docker

If you can run docker in your environment, the setup is very simple. 
It is all contained in the `Dockerfile`, which was gratefully provided by
Abdolhamid Pourghazi <pourgh01@ads.uni-passau.de> and 
Stefan Klessinger <stefan.klessinger@uni-passau.de> from
University of Passau, Germany.

From the main directory `qcc` (which contains the `Dockerfile`), simply run:

```
docker build -t qcc4cp:latest .
docker run -t -d --name qcc4cp qcc4cp:latest
docker exec -it qcc4cp /bin/bash
```

And then run all the algorithms (which are all on `.py` files in the `qcc/src` directory, with:

```
for algo in `ls -1 *py | sed s@.py@@g`
    do
       bazel run $algo
    done
```


