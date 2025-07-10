OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];

// moment 0
h q[6];
h q[7];
h q[2];

// moment 1
h q[1];
h q[4];
cz q[6], q[2];

// moment 2
h q[0];
cz q[6], q[1];

// moment 3
cz q[7], q[1];
cx q[5], q[0];

// moment 4
cz q[1], q[4];
cx q[0], q[5];

// moment 5
cz q[2], q[4];
x q[1];
cx q[5], q[0];

// moment 6
cz q[5], q[4];

// moment 7
cz q[7], q[5];
x q[4];

// moment 8
x q[6];
x q[7];
x q[5];
x q[2];

// measurement
measure q[6]->c[0];
measure q[7]->c[1];
measure q[5]->c[2];
measure q[2]->c[3];
measure q[1]->c[4];
measure q[4]->c[5];
