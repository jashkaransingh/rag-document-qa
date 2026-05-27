# Mark VII Technical Specifications

The Mark VII is Acme's seventh-generation industrial robot chassis, designed
for general-purpose manufacturing automation in spaces with mixed human and
machine workflows.

## Physical

Dimensions: 1.2m tall, 80cm wide, 70cm deep when arms are retracted.
Mass: 142 kilograms.
Payload capacity: 120 kilograms (continuous), 180 kg peak.
Reach: 1.8 meter sphere from the base.
Repeatability: plus or minus 0.05 millimeters at the end effector.

## Power

Standard configuration runs on 48VDC from a wall outlet or onboard battery.
Battery life is 6 hours under typical industrial workload, 4 hours under
sustained peak load.
Hot-swap capable, two slots, no power down required for battery exchange.

## Compute and sensing

Onboard compute: Nvidia Jetson AGX Orin, 64GB unified memory.
Cameras: 4 RGB-D cameras (front, back, left, right), each capturing at
1080p 60fps with depth at 30fps.
LiDAR: 360 degree 32-channel rotating scanner, 30 Hz refresh rate.
Force sensing: 6-axis force/torque sensor at each wrist.

## Software

Runs AcmeOS 4.2 or later. Compatible with ROS 2 Humble and Iron.
Motion planning stack supports both classical RRT-Connect and learned
diffusion-based policies for cluttered environments.
Vision pipeline runs at 90 fps end to end.

## Safety

Compliant with ISO 10218-1:2011 and ISO/TS 15066:2016 for collaborative
operation alongside humans.
Emergency stop is hardwired across all joints and cannot be overridden by
software.
Force-limited motion mode reduces all axes to a maximum of 25 newtons of
contact force when a human is detected within 1.5 meters.
