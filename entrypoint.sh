#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source ${ROS_WS}/install/setup.bash

case "${DEPLOYMENT_ENV:-local}" in
  robot)
    export CYCLONEDDS_URI=${ROS_WS}/config/cyclonedds_robot.xml
    ;;
  local)
    export CYCLONEDDS_URI=${ROS_WS}/config/cyclonedds_local.xml
    ;;
  *)
    echo "DEPLOYMENT_ENV inconnu: '${DEPLOYMENT_ENV}' (attendu: robot|local). Utilisation de local par défaut." >&2
    export CYCLONEDDS_URI=${ROS_WS}/config/cyclonedds_local.xml
    ;;
esac

echo "DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-local} -> CYCLONEDDS_URI=${CYCLONEDDS_URI}"

exec "$@"