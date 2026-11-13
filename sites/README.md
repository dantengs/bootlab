# sites

A site is the configuration of one cluster: `sites/<name>/site.toml` and `sites/<name>/setup.sh`.

You do not write it by hand. Generate it from a probe of your cluster:

```bash
bin/bootlab probe plan --sif <image> --out probe --submit     # one small job per node type
bin/bootlab probe merge probe                                 # when the jobs are done
touch mysite.hand.toml                                        # partition, time limit: may stay empty
bin/bootlab site gen --probe probe/probe.json --hand mysite.hand.toml --out sites/mysite
echo 'site = "mysite"' >> bootlab.toml
```

`_template/site.toml` explains every key, for reading or for a site written by hand. Measured resource data
(`bench/<node type>.json`) is optional; without it a run uses every physical core of one node. See
[docs/en/tutorial.md](../docs/en/tutorial.md) sections 2, 3 and 9 ([中文](../docs/zh/tutorial.md)).

站点就是一个集群的配置。不用手写，按上面的命令探测后自动生成；`_template/site.toml` 逐项说明各个键。
