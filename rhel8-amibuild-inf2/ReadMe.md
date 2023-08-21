On AMI name: `RHEL-8.8.0_HVM-20230802-x86_64-64-Hourly2-GP2`



```
chmod +x ./build_drivers.sh
./build_drivers.sh
export PATH=/opt/aws/neuron/bin:$PATH
```
OR
```
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ashivadi/compile-llm-inf2/master/rhel8-amibuild-inf2/build_drivers.sh)"
```

Once done, test with:

```
neuron-ls # or /opt/aws/neuron/bin/neuron-ls if there are PATH issues
```
