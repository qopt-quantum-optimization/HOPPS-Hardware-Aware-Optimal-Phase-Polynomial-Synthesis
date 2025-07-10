OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];

// moment 0
h q[5];
h q[2];
h q[7];
h q[1];
h q[0];
h q[3];

// moment 1
cz q[5], q[7];

// moment 2
cx q[3], q[7];

// moment 3
h q[4];
cz q[5], q[0];
cx q[7], q[3];

// moment 4
cz q[5], q[4];
cx q[3], q[7];

// moment 5
cz q[5], q[7];
cz q[3], q[0];

// moment 6
cx q[5], q[0];

// moment 7
cx q[2], q[4];
cx q[0], q[5];

// moment 8
cx q[4], q[2];
cx q[5], q[0];

// moment 9
cz q[5], q[7];
x q[0];
cx q[2], q[4];

// moment 10
cz q[4], q[5];
x q[3];

// moment 11
cz q[4], q[2];
x q[5];

// moment 12
cz q[4], q[1];

// moment 13
cz q[1], q[7];
x q[2];

// moment 14
x q[4];
x q[1];
x q[7];

// measurement
measure q[0]->c[0];
measure q[4]->c[1];
measure q[3]->c[2];
measure q[1]->c[3];
measure q[5]->c[4];
measure q[2]->c[5];
measure q[7]->c[6];
