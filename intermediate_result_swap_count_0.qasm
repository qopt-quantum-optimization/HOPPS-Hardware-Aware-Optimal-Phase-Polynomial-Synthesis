OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];

// moment 0
h q[1];
h q[3];
h q[4];
h q[7];
h q[5];

// moment 1
cz q[1], q[4];

// moment 2
h q[0];
cz q[1], q[7];

// moment 3
cz q[3], q[7];

// moment 4
cz q[7], q[5];

// moment 5
cz q[4], q[5];

// moment 6
cz q[0], q[5];
x q[1];
x q[4];

// moment 7
cz q[3], q[0];
x q[7];

// moment 8
x q[3];
x q[0];
x q[5];

// measurement
measure q[1]->c[0];
measure q[3]->c[1];
measure q[0]->c[2];
measure q[4]->c[3];
measure q[7]->c[4];
measure q[5]->c[5];
