OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];

// moment 0
h q[8];
h q[10];
h q[6];

// moment 1
h q[5];
cz q[6], q[8];
cx q[1], q[2];

// moment 2
cx q[2], q[1];
cx q[5], q[6];
cx q[10], q[4];

// moment 3
h q[9];
cx q[1], q[2];
cx q[6], q[5];
cx q[4], q[10];

// moment 4
cz q[9], q[8];
cx q[5], q[6];
cx q[10], q[4];

// moment 5
cz q[9], q[5];

// moment 6
cz q[6], q[8];
cx q[10], q[9];

// moment 7
cz q[6], q[5];
cx q[9], q[10];

// moment 8
cz q[5], q[4];
x q[8];
x q[6];
cx q[10], q[9];

// moment 9
cz q[10], q[4];
x q[5];

// moment 10
x q[4];
x q[10];

// measurement
measure q[8]->c[0];
measure q[4]->c[1];
measure q[5]->c[2];
measure q[10]->c[3];
measure q[6]->c[4];
