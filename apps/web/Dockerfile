FROM node:20-alpine AS build
WORKDIR /app
COPY apps/web/package.json /app/package.json
COPY apps/web /app
RUN npm install && npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
