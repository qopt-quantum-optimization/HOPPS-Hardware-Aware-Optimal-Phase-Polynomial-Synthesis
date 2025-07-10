OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];

// moment 0
h q[4];
h q[1];
h q[5];
h q[7];

// moment 1
cz q[4], q[1];

// moment 2
cz q[4], q[5];
cx q[7], q[1];

// moment 3
h q[3];
cx q[1], q[7];

// moment 4
cx q[7], q[1];

// moment 5
cz q[7], q[5];

// moment 6
cz q[4], q[1];
cx q[0], q[5];

// moment 7
cz q[7], q[1];
cx q[5], q[0];

// moment 8
cz q[3], q[7];
x q[4];
cx q[0], q[5];

// moment 9
cz q[3], q[0];

// moment 10
x q[3];
x q[7];
x q[0];
x q[1];

// measurement
measure q[4]->c[0];
measure q[3]->c[1];
measure q[7]->c[2];
measure q[0]->c[3];
measure q[1]->c[4];
