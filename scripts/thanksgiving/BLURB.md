# Why it's useful

King Crab is a playful demo of an assistive tabletop robot. A small, low-cost walking robot
lives on the table and passes things to the people who can't easily reach them: an older
relative at the far end, someone in a wheelchair, a child who can't stretch across a crowded
table. It walks to an item, takes it in its claw, carries it level and offers it to the person
who needs it. In this simulation the open-source Jumper robot runs its real, shipped
controller and neural-network walking and grasping policies on MuJoCo physics, so the same
software stack can run on the physical robot. Only the high-level "who needs what" choices and
the grip itself are scripted. The point of the story is the point of the robot: the most
useful thing a helper can pass is whatever keeps everyone at the table together.
