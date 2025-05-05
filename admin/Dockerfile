# Use an official Node runtime as a parent image
# Choose a version compatible with your Next.js project (e.g., 18 or 20)
FROM node:18-alpine

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./
# Or if using yarn:
# COPY package.json yarn.lock ./

# Install dependencies
RUN npm install
# Or if using yarn:
# RUN yarn install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Next.js builds the app in development mode automatically,
# but for production you might need: RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Command to run the development server
# Host 0.0.0.0 needed to access from outside the container
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0"]
# Or if using yarn:
# CMD ["yarn", "dev", "-H", "0.0.0.0"]