import type React from "react"
import Svg, { Path } from "react-native-svg"

interface CloudProps {
  width: number
  height: number
  stroke: string
  fill?: string
}

const Cloud: React.FC<CloudProps> = ({ width, height, stroke, fill = "none" }) => {
  return (
    <Svg
      width={width}
      height={height}
      viewBox="0 0 24 24"
      fill={fill}
      stroke={stroke}
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <Path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
    </Svg>
  )
}

export default Cloud
